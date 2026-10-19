"""CMI predictor export, modality-aware evaluation, and submission evidence."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mlflow import MlflowClient
from prefect import task

from dsio.config.components import resolve_component
from dsio.data.store import SignalStore
from dsio.eval import evaluate
from dsio.experimental.data import collate_arrays
from dsio.experimental.inference import MulticlassOutput
from dsio.inference import build_predictor, log_predictor, predict, require_checkpoint_lineage
from dsio.tracking import attempt, record_provenance
from dsio.train.artifacts import ArtifactRef, save_artifact
from reference_projects.kaggle.child_mind.components import INPUTS, OUTPUT
from reference_projects.kaggle.child_mind.data import SENSOR_PRESENT_INDEX, SENSOR_VALUE_SLICE


@task(persist_result=False)
def export(
    data: dict[str, Any], split: dict[str, Any], training: dict[str, Any], experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        reference = ArtifactRef.model_validate(training["checkpoint"])
        components = require_checkpoint_lineage(
            reference,
            training_run_id=training["train_run_id"],
            training_identity=training["identity"],
            dataset_digest=data["dataset_digest"],
            split_digest=split["split_digest"],
            components=("model", "preprocessor"),
        )
        model = resolve_component(components["model"], expected=torch.nn.Module)
        preprocessor = resolve_component(components["preprocessor"], expected=torch.nn.Module)
        sample_id = next(iter(split["assignments"]["validate"]))
        inputs = collate_arrays(INPUTS, SignalStore(data["store_path"]), [sample_id])
        output = resolve_component(OUTPUT, expected=MulticlassOutput)
        identity = record_provenance(
            run.info.run_id,
            {
                "checkpoint_digest": reference.digest,
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "mode": training["mode"],
                "export_form": "pyfunc",
            },
            components={
                "builder": "dsio.inference.predictor:build_predictor",
                "model": components["model"],
                "preprocessor": components["preprocessor"],
                "normalizer": OUTPUT,
                "validator": {
                    "reference": (
                        "dsio.experimental.inference.outputs.multiclass:MulticlassValidator"
                    ),
                    "parameters": OUTPUT["parameters"],
                },
            },
        )
        example = {
            "sample_id": inputs["sample_id"].tolist(),
            "x": torch.from_numpy(inputs["x"]),
        }
        predictor = build_predictor(
            reference,
            model=model,
            preprocessor=preprocessor,
            normalizer=output,
            validator=output.validator,
            input_example=example,
        )
        info = log_predictor(
            predictor,
            run_id=run.info.run_id,
            input_example=example,
            forms=("pyfunc",),
            name=f"child-mind-{training['mode']}",
        )["pyfunc"]
        return {
            "export_run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "checkpoint_digest": reference.digest,
            "mode": training["mode"],
            "identity": identity,
        }


@task(persist_result=False)
def evaluate_model(
    data: dict[str, Any], split: dict[str, Any], exported: dict[str, Any], experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        sample_ids = list(split["assignments"]["validate"])
        store = SignalStore(data["store_path"])
        targets = np.asarray(
            [int(store.read_sample(value)["attrs"]["target"]) for value in sample_ids],
            dtype=np.int64,
        )
        inputs = collate_arrays(INPUTS, store, sample_ids)
        present = inputs["x"][:, 0, SENSOR_PRESENT_INDEX].astype(bool)
        present_ids = inputs["sample_id"][present].tolist()
        missing_ids = inputs["sample_id"][~present].tolist()
        if not present_ids or not missing_ids:
            raise ValueError("CMI validation requires both present and missing sensor modalities")
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "mode": exported["mode"],
                "metrics": [
                    "accuracy",
                    "quadratic_weighted_kappa",
                    "qwk.sensor_present",
                    "qwk.sensor_missing",
                    "qwk.sensor_ablated",
                    "qwk.sensor_ablation_delta",
                ],
                "sample_ids": sample_ids,
                "sensor_present_sample_ids": present_ids,
                "sensor_missing_sample_ids": missing_ids,
                "ablation_sample_ids": present_ids,
            },
            components={"evaluation": "dsio.eval.execution:evaluate"},
        )
        metrics = evaluate(
            run_id=run.info.run_id,
            model_uri=exported["model_uri"],
            dataset_run_id=split["split_run_id"],
            inputs=inputs,
            targets=targets,
            metrics=("accuracy", "quadratic_weighted_kappa"),
            subgroups={"sensor_present": present, "sensor_missing": ~present},
            subgroup_metrics={"qwk": "quadratic_weighted_kappa"},
            ablations={
                "sensor": {
                    "input_field": "x",
                    "slice": [SENSOR_VALUE_SLICE.start, SENSOR_PRESENT_INDEX + 1],
                    "subgroup": "sensor_present",
                    "fill": 0.0,
                    "metrics": {"qwk": "quadratic_weighted_kappa"},
                }
            },
        )
        return {
            "evaluation_run_id": run.info.run_id,
            "metrics": metrics,
            "sensor_present_sample_ids": present_ids,
            "ablation_sample_ids": present_ids,
            "identity": identity,
        }


@task(persist_result=False)
def infer_and_submit(
    data: dict[str, Any], exported: dict[str, Any], workspace: str, experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        test_ids = list(data["test_ids"])
        test_sample_ids = list(data["test_sample_ids"])
        outputs = predict(
            exported["model_uri"],
            collate_arrays(INPUTS, SignalStore(data["store_path"]), test_sample_ids),
        )
        values = np.asarray(outputs["prediction"], dtype=np.int64).reshape(-1).tolist()
        if any(value not in range(4) for value in values):
            raise ValueError("CMI submission predictions must be integers in [0, 3]")
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "mode": exported["mode"],
                "sample_ids": test_sample_ids,
                "submission_ids": test_ids,
            },
            components={"inference": "dsio.inference.loading:predict"},
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["id", "sii"])
        writer.writerows(zip(test_ids, values, strict=True))
        payload = buffer.getvalue().encode()
        path = Path(workspace) / run.info.run_id / f"submission-{exported['mode']}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        submission = save_artifact(payload, run_id=run.info.run_id, name="submission")
        MlflowClient().log_dict(
            run.info.run_id, submission.model_dump(mode="json"), "outputs/submission.json"
        )
        return {
            "inference_run_id": run.info.run_id,
            "prediction": values,
            "prediction_count": len(values),
            "submission_path": str(path),
            "submission_bytes": payload,
            "submission_digest": submission.digest,
            "identity": identity,
        }
