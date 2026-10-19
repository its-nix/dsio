from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.golden import assert_golden_metrics
from tests.replay import assert_same_identities
from torch.utils.data import default_collate

from dsio.data.loading import DsioDataModule
from dsio.experimental.model import InstanceStandardize, Standardize, SupervisedObjective
from dsio.model.module import DsioModule


class _LegacySequence(torch.nn.Module):
    def __init__(
        self,
        *,
        window_length: int,
        sensor_features: int,
        tabular_center: list[float],
        tabular_scale: list[float],
        hidden: int,
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.sensor_features = sensor_features
        self.tabular_standardize = Standardize(tabular_center, tabular_scale, observed_only=True)
        self.signal_standardize = InstanceStandardize(eps=1e-6, nan_policy="omit")
        self.tabular_encoder = torch.nn.Sequential(torch.nn.Linear(116, hidden), torch.nn.GELU())
        self.temporal_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(sensor_features * 2, hidden, 9, 4, 4),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden, hidden, 9, 4, 4),
            torch.nn.GELU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2 + 1, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tabular_values, tabular_mask = x[:, :58], x[:, 58:116]
        signal = x[:, 116:-1].reshape(-1, self.window_length, self.sensor_features)
        present = x[:, -1:]
        absent = present[:, None, :].eq(0)
        observed = (torch.isfinite(signal) & ~absent).to(x.dtype)
        tabular = self.tabular_standardize(tabular_values, tabular_mask)
        tabular_encoded = self.tabular_encoder(torch.cat((tabular, tabular_mask), 1))
        signal = self.signal_standardize(
            torch.where(absent, torch.full_like(signal, float("nan")), signal).transpose(1, 2)
        ).transpose(1, 2)
        temporal = self.temporal_encoder(torch.cat((signal, observed), 2).transpose(1, 2))
        return self.head(
            torch.cat((tabular_encoded, temporal.mean(dim=2) * present, present), dim=1)
        )


def test_sequence_corpus_preserves_raw_rows_missing_modalities_and_participants(
    child_mind_data: Path, tmp_path: Path
) -> None:
    from reference_projects.kaggle.child_mind.sequence.data import (
        RAW_SENSOR_FEATURES,
        sequence_examples,
        stage_sequence_store,
    )

    store = stage_sequence_store(child_mind_data, tmp_path / "sequence-store", window_length=4)
    examples = sequence_examples(store, window_length=4)

    assert store.channels == len(RAW_SENSOR_FEATURES)
    assert len(store.entities) == 48
    assert store.n_rows == 240  # 144 real sensor rows + 24 explicit four-row placeholders.
    assert len(examples) == 72  # Two windows per present participant, one per missing one.
    assert len(set(examples.groups.tolist())) == 48
    assert set(examples.attribute("sensor_present").tolist()) == {False, True}
    assert set(examples.attribute("label").tolist()) == {0, 1, 2, 3}

    missing = next(entity for entity in store.entities if not entity.attrs["sensor_present"])
    missing_values = np.asarray(store.read_sample(missing.entity_id)["data"])
    assert missing_values.shape == (4, len(RAW_SENSOR_FEATURES))
    assert np.count_nonzero(missing_values) == 0

    covered = np.zeros(store.n_rows, dtype=bool)
    for start in examples.index.starts:
        covered[int(start) : int(start) + 4] = True
    assert bool(covered.all())


def test_sequence_components_train_on_raw_windows_without_participant_length_bias(
    child_mind_data: Path, tmp_path: Path
) -> None:
    from reference_projects.kaggle.child_mind.sequence.components import (
        DATASET,
        OBJECTIVE,
        model_config,
        model_runtime_parameters,
    )
    from reference_projects.kaggle.child_mind.sequence.data import (
        RAW_SENSOR_FEATURES,
        sequence_examples,
        stage_sequence_store,
    )

    from dsio.config.components import resolve_component
    from dsio.experimental.data import WindowItems

    store = stage_sequence_store(child_mind_data, tmp_path / "sequence-store", window_length=4)
    examples = sequence_examples(store, window_length=4)
    sample_ids = examples.sample_ids.tolist()
    counts = Counter(str(group) for group in examples.groups)
    sample_weights = {
        sample_id: len(sample_ids) / (len(counts) * counts[str(group)])
        for sample_id, group in zip(sample_ids, examples.groups, strict=True)
    }
    dataset = resolve_component(
        DATASET,
        fitted_sample_weights=sample_weights,
        expected=WindowItems,
    )(store, examples, sample_ids)

    participant_weights: dict[str, float] = {}
    present_item = None
    missing_item = None
    for position in range(len(dataset)):
        item = dataset[position]
        participant = str(item["group"])
        participant_weights[participant] = participant_weights.get(participant, 0.0) + float(
            item["sample_weight"]
        )
        present_item = item if bool(item["x"][-1]) and present_item is None else present_item
        missing_item = item if not bool(item["x"][-1]) and missing_item is None else missing_item

    assert len(set(round(value, 6) for value in participant_weights.values())) == 1
    assert present_item is not None and missing_item is not None
    assert present_item["x"].shape == missing_item["x"].shape
    assert float(present_item["x"][-1]) == 1.0
    assert float(missing_item["x"][-1]) == 0.0

    batch = default_collate([present_item, missing_item])
    model_parameters = {
        "window_length": 4,
        "sensor_features": len(RAW_SENSOR_FEATURES),
        "tabular_center": [0.0] * 58,
        "tabular_scale": [1.0] * 58,
        "hidden": 8,
    }
    torch.manual_seed(29)
    model = _LegacySequence(**model_parameters)
    torch.manual_seed(29)
    composed = resolve_component(
        model_config(window_length=4, sensor_features=len(RAW_SENSOR_FEATURES), hidden=8),
        **model_runtime_parameters(
            tabular_center=model_parameters["tabular_center"],
            tabular_scale=model_parameters["tabular_scale"],
        ),
        expected=torch.nn.Module,
    )
    for expected, actual in zip(model.parameters(), composed.parameters(), strict=True):
        assert expected.shape == actual.shape
        assert torch.equal(expected, actual)
    assert torch.equal(composed(batch["x"]), model(batch["x"]))
    assert torch.equal(composed(batch["x"].double()), model(batch["x"]))
    objective = resolve_component(
        OBJECTIVE,
        loss_parameters={"weight": [1.0] * 4},
        expected=SupervisedObjective,
    )
    values = objective(model, batch, "train")

    assert values["loss"].ndim == 0
    assert bool(torch.isfinite(values["loss"]))
    assert 0.0 <= float(values["accuracy"]) <= 1.0


def test_sequence_objective_is_invariant_to_batch_partitioning() -> None:
    from reference_projects.kaggle.child_mind.sequence.components import OBJECTIVE

    from dsio.config.components import resolve_component

    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    batch = {
        "x": logits,
        "y": torch.tensor([0, 1, 2]),
        "sample_weight": torch.tensor([1.5, 0.75, 0.75]),
    }
    class_weights = torch.tensor([0.5, 1.0, 1.5, 2.0])
    whole_objective = resolve_component(
        OBJECTIVE, loss_parameters={"weight": class_weights}, expected=SupervisedObjective
    )
    partitioned_objective = resolve_component(
        OBJECTIVE, loss_parameters={"weight": class_weights}, expected=SupervisedObjective
    )

    whole = whole_objective(torch.nn.Identity(), batch, "train")
    parts = [
        partitioned_objective(
            torch.nn.Identity(), {name: value[:1] for name, value in batch.items()}, "train"
        ),
        partitioned_objective(
            torch.nn.Identity(), {name: value[1:] for name, value in batch.items()}, "train"
        ),
    ]
    losses = torch.nn.functional.cross_entropy(
        logits, batch["y"], weight=class_weights, reduction="none"
    )

    partitioned_loss = (parts[0]["loss"] + 2 * parts[1]["loss"]) / 3
    assert torch.equal(whole["loss"], (losses * batch["sample_weight"]).mean())
    assert partitioned_loss == pytest.approx(float(whole["loss"]))
    partitioned_accuracy = (parts[0]["accuracy"] + 2 * parts[1]["accuracy"]) / 3
    assert whole["accuracy"] == pytest.approx(0.75)
    assert partitioned_accuracy == pytest.approx(float(whole["accuracy"]))


def test_sequence_flow_trains_raw_windows_and_scores_participants(
    child_mind_data: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    import mlflow.pyfunc
    from lightning import Trainer
    from mlflow import MlflowClient
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.child_mind.sequence.components import DATASET
    from reference_projects.kaggle.child_mind.sequence.flow import child_mind_sequence_flow

    observed: list[tuple[type[object], type[object]]] = []
    loaded_model_uris: list[str] = []
    original = Trainer.fit
    original_load_model = mlflow.pyfunc.load_model

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        observed.append((type(model), type(kwargs["datamodule"])))
        return original(trainer, model, *args, **kwargs)

    def record_model_load(model_uri: str, *args: object, **kwargs: object) -> object:
        loaded_model_uris.append(model_uri)
        return original_load_model(model_uri, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", record)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", record_model_load)
    with prefect_test_harness():
        result = child_mind_sequence_flow(
            str(child_mind_data),
            str(tmp_path / "work"),
            seed=17,
            window_length=4,
            max_epochs=1,
            accelerator="cpu",
            calibrate=False,
        )

    assert observed == [(DsioModule, DsioDataModule)]
    assert result["participant_count"] == 48
    assert result["sensor_present_participants"] == 24
    assert result["sensor_missing_participants"] == 24
    assert result["raw_sensor_rows"] == 144
    assert result["window_count"] == 72
    assert set(result["split_groups"]["train"]).isdisjoint(result["split_groups"]["validate"])
    assert set(result["metrics"]) == {
        "accuracy",
        "quadratic_weighted_kappa",
        "qwk.sensor_present",
        "qwk.sensor_missing",
        "qwk.sensor_ablated",
        "qwk.sensor_ablation_delta",
    }
    assert all(np.isfinite(value) for value in result["metrics"].values())
    assert result["model_uri"].startswith("models:/m-")
    assert result["model_uri"] in loaded_model_uris
    assert len(result["checkpoint_digest"]) == 64

    training = MlflowClient().get_run(result["train_run_id"])
    assert training.data.metrics["train/loss_epoch"] >= 0
    assert training.data.metrics["val/loss"] >= 0
    assert 0.0 <= training.data.metrics["train/accuracy"] <= 1.0
    assert 0.0 <= training.data.metrics["val/accuracy"] <= 1.0
    provenance_path = MlflowClient().download_artifacts(
        result["train_run_id"], "provenance.json", str(tmp_path / "provenance")
    )
    provenance = json.loads(Path(provenance_path).read_text())
    assert provenance["components"]["dataset_factory"] == DATASET
    fitted = provenance["configuration"]["standardization"]
    assert fitted["role"] == "train"
    assert fitted["partition"] == 0
    assert fitted["source"] == "entity_attribute"
    assert fitted["attribute"] == "tabular"
    assert fitted["observation"] == "mask"
    assert len(fitted["resolved_sample_ids"]) < len(fitted["sample_ids"])
    assert len(fitted["mean"]) == len(fitted["scale"]) == 58
    artifact = json.loads(
        Path(
            MlflowClient().download_artifacts(result["train_run_id"], fitted["artifact"])
        ).read_text()
    )
    assert artifact == {key: value for key, value in fitted.items() if key != "artifact"}
    weights = provenance["configuration"]["weights"]
    assert set(weights) == {"class", "group"}
    assert weights["class"]["source"] == "entity_attribute"
    assert len(weights["class"]["resolved_sample_ids"]) < len(weights["class"]["sample_ids"])
    assert weights["group"]["sample_ids"] == list(weights["group"]["sample_weights"])
    assert sum(weights["group"]["sample_weights"].values()) / len(
        weights["group"]["sample_weights"]
    ) == pytest.approx(1.0)
    for evidence in weights.values():
        weight_artifact = json.loads(
            Path(
                MlflowClient().download_artifacts(result["train_run_id"], evidence["artifact"])
            ).read_text()
        )
        assert weight_artifact == {
            key: value for key, value in evidence.items() if key != "artifact"
        }
    for run_key in ("export_run_id", "evaluation_run_id"):
        stage_provenance_path = MlflowClient().download_artifacts(
            result[run_key], "provenance.json", str(tmp_path / f"{run_key}-provenance")
        )
        stage_provenance = json.loads(Path(stage_provenance_path).read_text())
        assert stage_provenance["components"]["dataset_factory"] == DATASET
    evaluation = MlflowClient().get_run(result["evaluation_run_id"])
    for name, value in result["metrics"].items():
        assert evaluation.data.metrics[name] == pytest.approx(value)
    assert evaluation.inputs is not None
    assert [value.model_id for value in evaluation.inputs.model_inputs] == [
        result["model_uri"].removeprefix("models:/")
    ]
    predictions_path = MlflowClient().download_artifacts(
        result["evaluation_run_id"], "outputs/participant-predictions.json"
    )
    predictions = json.loads(Path(predictions_path).read_text())
    assert len(predictions["sensor_ablated_prediction"]) == predictions["participant_count"]


def test_sequence_flow_replays_data_split_weights_and_metrics(
    child_mind_data: Path,
    tmp_path: Path,
    kaggle_services: None,
) -> None:
    del kaggle_services
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.child_mind.sequence.flow import child_mind_sequence_flow

    parameters = {
        "seed": 17,
        "window_length": 4,
        "max_epochs": 1,
        "accelerator": "cpu",
        "calibrate": False,
    }
    with prefect_test_harness():
        first = child_mind_sequence_flow(str(child_mind_data), str(tmp_path / "work"), **parameters)
        replay = child_mind_sequence_flow(
            str(child_mind_data), str(tmp_path / "work"), **parameters
        )

    assert first["dataset_digest"] == replay["dataset_digest"]
    assert first["split_digest"] == replay["split_digest"]
    assert first["assignments"] == replay["assignments"]
    assert first["checkpoint_digest"] == replay["checkpoint_digest"]
    assert first["metrics"] == replay["metrics"]
    assert_same_identities(first, replay)
    assert_golden_metrics("child_mind_sequence", first["metrics"])
    assert first["train_run_id"] != replay["train_run_id"]
    assert first["model_uri"] != replay["model_uri"]
