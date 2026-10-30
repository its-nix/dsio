"""Canonical DSIO Lightning training for both CMI configurations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import mlflow
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from prefect import task

from dsio.config.components import ComponentConfig, resolve_component
from dsio.data.loading import DsioDataModule
from dsio.data.store import SignalStore
from dsio.experimental import data as experimental_data
from dsio.experimental.data import fit_class_weights, record_fitted
from dsio.experimental.model import SupervisedObjective
from dsio.model.module import DsioModule
from dsio.tracking import attempt, load_split_evidence, record_provenance
from dsio.train.artifacts import save_artifact
from dsio.train.capabilities import (
    check_requested_capabilities,
    log_capabilities,
    resolve_training_capabilities,
)
from dsio.train.trainer import TrainerConfig, build_callbacks, build_trainer
from reference_projects.kaggle.child_mind.components import (
    CLASSES,
    DATASET,
    OBJECTIVE,
    model_config,
    model_runtime_parameters,
)
from reference_projects.kaggle.child_mind.data import (
    SENSOR_FEATURES,
    SENSOR_MASK_SLICE,
    SENSOR_VALUE_SLICE,
    TABULAR_FEATURES,
    TABULAR_MASK_SLICE,
    TABULAR_VALUE_SLICE,
)
from reference_projects.kaggle.child_mind.tasks.data import labelled_examples

Mode = Literal["tabular", "fused"]
TRAINER = TrainerConfig(
    max_epochs=2,
    accelerator="cpu",
    devices=1,
    deterministic=True,
    checkpoint=False,
    log_every_n_steps=1,
    limit_val_batches=1.0,
    num_sanity_val_steps=0,
)
ROLES = {"train": "train", "validate": "validate"}
SHUFFLE = {"train": True, "validate": False, "test": False, "predict": False}
DROP_LAST = {"train": False, "validate": False, "test": False, "predict": False}
BATCH_SIZE = 16
NUM_WORKERS = 0
OPTIMIZER_PARAMETERS = {"lr": 0.003, "weight_decay": 0.0001}
PREPROCESSOR: ComponentConfig = {"reference": "torch.nn:Identity", "parameters": {}}


@task(persist_result=False)
def train(
    data: dict[str, Any],
    split: dict[str, Any],
    experiment_id: str,
    seed: int,
    mode: Mode,
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        if mode not in ("tabular", "fused"):
            raise ValueError(f"unsupported CMI mode {mode!r}")
        check_requested_capabilities(TRAINER)
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = load_split_evidence(
            split["split_uri"], examples, consumer_run_id=run.info.run_id
        )
        tabular = experimental_data.fit_standardization(
            store,
            manifest,
            partition=0,
            value_columns=range(TABULAR_VALUE_SLICE.start, TABULAR_VALUE_SLICE.stop),
            observed_columns=range(TABULAR_MASK_SLICE.start, TABULAR_MASK_SLICE.stop),
            feature_names=[f"tabular_{index}" for index in range(TABULAR_FEATURES)],
            minimum_scale=1e-6,
            reduction="sum_squares",
        )
        sensor = experimental_data.fit_standardization(
            store,
            manifest,
            partition=0,
            value_columns=range(SENSOR_VALUE_SLICE.start, SENSOR_VALUE_SLICE.stop),
            observed_columns=range(SENSOR_MASK_SLICE.start, SENSOR_MASK_SLICE.stop),
            feature_names=[f"sensor_{index}" for index in range(SENSOR_FEATURES)],
            minimum_scale=1e-6,
            reduction="sum_squares",
        )
        class_weights = fit_class_weights(store, manifest, CLASSES, "target", partition=0)
        weights = class_weights["weights"]
        tabular["artifact"] = record_fitted(run.info.run_id, "tabular-standardization", tabular)
        sensor["artifact"] = record_fitted(run.info.run_id, "sensor-standardization", sensor)
        fitted_evidence = {"tabular": tabular, "sensor": sensor}
        class_weights["artifact"] = record_fitted(run.info.run_id, "class-weights", class_weights)
        weight_evidence = {"class": class_weights}
        statistics = {
            "tabular_center": tabular["mean"],
            "tabular_scale": tabular["scale"],
            "sensor_center": sensor["mean"],
            "sensor_scale": sensor["scale"],
        }
        configured_model = model_config(use_sensor=mode == "fused", hidden=32)
        runtime_model = model_runtime_parameters(**statistics)
        recorded_model_config = deepcopy(configured_model)
        recorded_model_config["parameters"].update(runtime_model)
        recorded_objective_config = deepcopy(OBJECTIVE)
        recorded_objective_config["parameters"]["loss_parameters"] = {"weight": weights}
        seed_everything(seed, workers=True, verbose=False)
        data_module = DsioDataModule(
            store,
            examples,
            manifest,
            fold=0,
            roles=ROLES,
            dataset_factory=resolve_component(DATASET, expected=experimental_data.StoredItems),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            seed=seed,
            shuffle=SHUFFLE,
            drop_last=DROP_LAST,
        )
        module = DsioModule(
            model=resolve_component(configured_model, **runtime_model, expected=torch.nn.Module),
            objective=resolve_component(
                OBJECTIVE,
                loss_parameters={"weight": weights},
                expected=SupervisedObjective,
            ),
            optimizer_factory=torch.optim.AdamW,
            optimizer_parameters=OPTIMIZER_PARAMETERS,
        )
        logger = MLFlowLogger(
            experiment_name="dsio-kaggle-child-mind",
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run.info.run_id,
            log_model=False,
        )
        directory = Path(data["store_path"]).parent / mode
        directory.mkdir(parents=True, exist_ok=True)
        trainer = build_trainer(
            TRAINER,
            directory,
            logger,
            build_callbacks(TRAINER, directory, has_validation=True),
        )
        execution = resolve_training_capabilities(trainer, requested=TRAINER)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "seed": seed,
                "fold": 0,
                "mode": mode,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "optimizer_parameters": OPTIMIZER_PARAMETERS,
                "roles": ROLES,
                "shuffle": SHUFFLE,
                "drop_last": DROP_LAST,
                "trainer": TRAINER.model_dump(mode="json"),
                "execution": execution,
                "standardization": fitted_evidence,
                "weights": weight_evidence,
            },
            components={
                "module": "dsio.model.module:DsioModule",
                "data_module": "dsio.data.loading.module:DsioDataModule",
                "dataset_factory": DATASET,
                "model": recorded_model_config,
                "objective": recorded_objective_config,
                "optimizer": "torch.optim:AdamW",
                "preprocessor": PREPROCESSOR,
            },
        )
        log_capabilities(logger, execution)
        trainer.fit(module, datamodule=data_module)
        checkpoint = directory / "child-mind.ckpt"
        trainer.save_checkpoint(checkpoint)
        reference = save_artifact(
            checkpoint.read_bytes(), run_id=run.info.run_id, name="checkpoint"
        )
        MlflowClient().log_dict(
            run.info.run_id, reference.model_dump(mode="json"), "outputs/checkpoint.json"
        )
        return {
            "train_run_id": run.info.run_id,
            "checkpoint": reference.model_dump(mode="json"),
            "mode": mode,
            "standardization": fitted_evidence,
            "weights": weight_evidence,
            "identity": identity,
        }
