from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from reference_projects.kaggle.child_mind.data import (
    SENSOR_MASK_SLICE,
    SENSOR_PRESENT_INDEX,
    SENSOR_VALUE_SLICE,
    TABULAR_MASK_SLICE,
    TABULAR_VALUE_SLICE,
)
from tests.golden import assert_golden_metrics
from tests.kaggle_portfolio.assertions import (
    assert_downstream_evidence,
    assert_execution_evidence,
    assert_replay_run_ids_differ,
)
from tests.replay import assert_same_identities

from dsio.config.components import resolve_component
from dsio.contracts import sha256_of_bytes
from dsio.data.loading import DsioDataModule
from dsio.experimental.inference import MulticlassOutput
from dsio.experimental.model import Standardize, SupervisedObjective
from dsio.model.module import DsioModule


class _LegacyFlat(torch.nn.Module):
    def __init__(self, parameters: dict[str, list[float]], hidden: int, use_sensor: bool) -> None:
        super().__init__()
        self.tabular_standardize = Standardize(
            parameters["tabular_center"], parameters["tabular_scale"], observed_only=True
        )
        self.sensor_standardize = Standardize(
            parameters["sensor_center"], parameters["sensor_scale"], observed_only=True
        )
        self.use_sensor = use_sensor
        self.tabular_encoder = torch.nn.Sequential(torch.nn.Linear(116, hidden), torch.nn.ReLU())
        self.sensor_encoder = torch.nn.Sequential(torch.nn.Linear(122, hidden), torch.nn.ReLU())
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2 + 1, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1).float()
        tabular_mask = x[:, TABULAR_MASK_SLICE]
        sensor_mask = x[:, SENSOR_MASK_SLICE]
        present = x[:, SENSOR_PRESENT_INDEX : SENSOR_PRESENT_INDEX + 1]
        tabular = self.tabular_standardize(x[:, TABULAR_VALUE_SLICE], tabular_mask)
        sensor = self.sensor_standardize(x[:, SENSOR_VALUE_SLICE], sensor_mask)
        tabular_encoded = self.tabular_encoder(torch.cat((tabular, tabular_mask), 1))
        sensor_encoded = self.sensor_encoder(torch.cat((sensor, sensor_mask), 1))
        if self.use_sensor:
            sensor_encoded = sensor_encoded * present
        else:
            sensor_encoded = torch.zeros_like(sensor_encoded)
            present = torch.zeros_like(present)
        return self.head(torch.cat((tabular_encoded, sensor_encoded, present), 1))


def test_child_mind_boundary_preserves_optional_modalities_and_excludes_leakage(
    child_mind_data: Path,
) -> None:
    from reference_projects.kaggle.child_mind.data import (
        PACKED_FEATURES,
        SAFE_FEATURE_COLUMNS,
        load_competition_data,
        pack_participant,
        summarize_partition,
    )

    loaded = load_competition_data(child_mind_data)

    assert len(loaded["train"]) == 48
    assert loaded["dropped_missing_targets"] == 4
    assert loaded["test_ids"] == [f"test-{index:03d}" for index in range(4)]
    assert not any(field.startswith("PCIAT-") or field == "sii" for field in SAFE_FEATURE_COLUMNS)

    with_series = loaded["train"][0]
    without_series = loaded["train"][4]
    summary = summarize_partition(Path(with_series["series_path"]), str(with_series["id"]))
    assert np.array_equal(
        summary.values,
        summarize_partition(Path(with_series["series_path"]), str(with_series["id"])).values,
    )
    present = pack_participant(with_series, summary)
    absent = pack_participant(without_series, None)
    assert present.shape == absent.shape == (1, PACKED_FEATURES)
    assert present[0, -1] == 1.0
    assert absent[0, -1] == 0.0
    assert np.isfinite(present).all()
    assert np.isfinite(absent).all()


def test_child_mind_boundary_rejects_finite_values_outside_float32_range(
    child_mind_data: Path,
) -> None:
    from reference_projects.kaggle.child_mind.data import (
        load_competition_data,
        pack_participant,
    )

    row = dict(load_competition_data(child_mind_data)["train"][0])
    row["Basic_Demos-Age"] = "3.5e38"

    with pytest.raises(ValueError, match="train-000.*Basic_Demos-Age.*float32"):
        pack_participant(row, None)


def test_child_mind_boundary_rejects_schema_drift_and_empty_sensor_partition(
    child_mind_data: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import (
        SENSOR_PARQUET_COLUMNS,
        load_competition_data,
        summarize_partition,
    )

    empty_id = "train-000"
    partition = child_mind_data / "series_train.parquet" / f"id={empty_id}" / "part-0.parquet"
    pq.write_table(
        pa.table(
            {
                name: pa.array([], type=pa.type_for_alias(field_type))
                for name, field_type in SENSOR_PARQUET_COLUMNS
            }
        ),
        partition,
    )
    with pytest.raises(ValueError, match=f"{empty_id}.*empty"):
        summarize_partition(partition, empty_id)

    path = child_mind_data / "test.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    write_fields = list(rows[0])[:-1]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=write_fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in write_fields} for row in rows])
    with pytest.raises(ValueError, match="test.csv columns"):
        load_competition_data(child_mind_data)


def test_child_mind_sensor_boundary_requires_exact_arrow_types(
    child_mind_data: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import (
        SENSOR_PARQUET_COLUMNS,
        summarize_partition,
    )

    participant_id = "train-000"
    partition = child_mind_data / "series_train.parquet" / f"id={participant_id}" / "part-0.parquet"
    columns = {
        name: pa.array(
            [0, 1],
            type=pa.float64() if name == "X" else pa.type_for_alias(field_type),
        )
        for name, field_type in SENSOR_PARQUET_COLUMNS
    }
    pq.write_table(pa.table(columns), partition)

    with pytest.raises(ValueError, match=f"{participant_id}.*X.*float32"):
        summarize_partition(partition, participant_id)


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([0, None, 2], "non-null"),
        ([0, 2, 1], "strictly increasing"),
        ([*range(65_536), 65_535], "strictly increasing"),
    ],
)
def test_child_mind_sensor_boundary_requires_valid_steps(
    child_mind_data: Path,
    steps: list[int | None],
    message: str,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import (
        SENSOR_PARQUET_COLUMNS,
        summarize_partition,
    )

    participant_id = "train-000"
    partition = child_mind_data / "series_train.parquet" / f"id={participant_id}" / "part-0.parquet"
    columns = {
        name: pa.array(
            steps if name == "step" else [0] * len(steps),
            type=pa.type_for_alias(field_type),
        )
        for name, field_type in SENSOR_PARQUET_COLUMNS
    }
    pq.write_table(pa.table(columns), partition)

    with pytest.raises(ValueError, match=f"{participant_id}.*step.*{message}"):
        summarize_partition(partition, participant_id)


def test_child_mind_sensor_summary_has_stable_variance_and_finite_float32_output(
    child_mind_data: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import (
        SENSOR_COLUMNS,
        summarize_partition,
    )

    participant_id = "train-000"
    partition = child_mind_data / "series_train.parquet" / f"id={participant_id}" / "part-0.parquet"
    table = pq.ParquetFile(partition).read()
    time_index = table.schema.get_field_index("time_of_day")
    base = 1_000_000_000_000
    table = table.set_column(
        time_index,
        "time_of_day",
        pa.array([base + offset for offset in (0, 2, 4, 6, 8)], type=pa.int64()),
    )
    float_index = table.schema.get_field_index("X")
    limit = np.finfo(np.float32).max
    table = table.set_column(
        float_index,
        "X",
        pa.array([limit, -limit, limit, -limit, 0], type=pa.float32()),
    )
    pq.write_table(table, partition)

    summary = summarize_partition(partition, participant_id)
    time_offset = 1 + SENSOR_COLUMNS.index("time_of_day") * 5
    assert summary.values[time_offset + 1] == pytest.approx(np.sqrt(8), rel=1e-6)
    assert np.isfinite(summary.values).all()


def test_child_mind_sensor_summary_rejects_infinite_source_values(
    child_mind_data: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reference_projects.kaggle.child_mind.data import summarize_partition

    participant_id = "train-000"
    partition = child_mind_data / "series_train.parquet" / f"id={participant_id}" / "part-0.parquet"
    table = pq.ParquetFile(partition).read()
    field_index = table.schema.get_field_index("X")
    table = table.set_column(
        field_index,
        "X",
        pa.array([0, 1, np.inf, 3, 4], type=pa.float32()),
    )
    pq.write_table(table, partition)

    with pytest.raises(ValueError, match=f"{participant_id}.*X.*infinity"):
        summarize_partition(partition, participant_id)


def test_child_mind_fusion_keeps_missing_sensor_explicit_and_ablation_local() -> None:
    from reference_projects.kaggle.child_mind.components import (
        OUTPUT,
        model_config,
        model_runtime_parameters,
    )
    from reference_projects.kaggle.child_mind.data import (
        PACKED_FEATURES,
        SENSOR_FEATURES,
        SENSOR_MASK_SLICE,
        SENSOR_PRESENT_INDEX,
        SENSOR_VALUE_SLICE,
        TABULAR_FEATURES,
    )

    batch = torch.zeros(2, 1, PACKED_FEATURES)
    batch[:, 0, :TABULAR_FEATURES] = torch.arange(TABULAR_FEATURES)
    batch[:, 0, TABULAR_FEATURES : 2 * TABULAR_FEATURES] = 1
    batch[0, 0, SENSOR_VALUE_SLICE] = 2
    batch[0, 0, SENSOR_MASK_SLICE] = 1
    batch[0, 0, SENSOR_PRESENT_INDEX] = 1
    batch[1, 0, SENSOR_VALUE_SLICE] = 999

    parameters = {
        "tabular_center": [0.0] * TABULAR_FEATURES,
        "tabular_scale": [1.0] * TABULAR_FEATURES,
        "sensor_center": [0.0] * SENSOR_FEATURES,
        "sensor_scale": [1.0] * SENSOR_FEATURES,
    }
    torch.manual_seed(41)
    model = _LegacyFlat(parameters, hidden=8, use_sensor=True)

    torch.manual_seed(41)
    composed = resolve_component(
        model_config(hidden=8), **model_runtime_parameters(**parameters), expected=torch.nn.Module
    )
    for expected, actual in zip(model.parameters(), composed.parameters(), strict=True):
        assert expected.shape == actual.shape
        assert torch.equal(expected, actual)
    logits = model(batch)
    assert torch.equal(composed(batch), logits)
    assert torch.equal(composed(batch.double()), model(batch.double()))
    assert torch.equal(composed(batch[:, 0]), logits)
    with pytest.raises(ValueError, match="fusion input"):
        composed(batch.transpose(1, 2))
    output = resolve_component(OUTPUT, expected=MulticlassOutput)
    outputs = output(logits)
    output.validator(outputs)
    assert logits.shape == (2, 4)
    assert outputs["prediction"].shape == (2,)
    assert outputs["probability"].shape == (2, 4)

    ablated = batch.clone()
    ablated[..., SENSOR_VALUE_SLICE.start : SENSOR_PRESENT_INDEX + 1] = 0

    torch.manual_seed(73)
    tabular_only = _LegacyFlat(parameters, hidden=8, use_sensor=False)
    torch.manual_seed(73)
    composed_tabular_only = resolve_component(
        model_config(hidden=8, use_sensor=False),
        **model_runtime_parameters(**parameters),
        expected=torch.nn.Module,
    )
    assert torch.equal(composed_tabular_only(batch), tabular_only(batch))
    assert torch.equal(tabular_only(batch), tabular_only(ablated))
    assert torch.equal(composed_tabular_only(batch), composed_tabular_only(ablated))
    with pytest.raises(ValueError, match="hidden"):
        model_config(hidden=0)


def test_child_mind_objective_matches_native_weighted_loss_and_accuracy() -> None:
    from reference_projects.kaggle.child_mind.components import OBJECTIVE

    logits = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    target = torch.tensor([0, 2, 1])
    class_weights = [0.5, 1.0, 1.5, 2.0]
    objective = resolve_component(
        OBJECTIVE,
        loss_parameters={"weight": class_weights},
        expected=SupervisedObjective,
    )

    values = objective(torch.nn.Identity(), {"x": logits, "y": target}, "train")
    expected_loss = torch.nn.functional.cross_entropy(
        logits, target, weight=torch.tensor(class_weights)
    )
    expected_accuracy = logits.argmax(dim=1).eq(target).float().mean()
    assert torch.equal(values["loss"], expected_loss)
    assert torch.equal(values["accuracy"], expected_accuracy)


def test_child_mind_fusion_ignores_masked_nonfinite_values_but_rejects_observed_ones() -> None:
    from reference_projects.kaggle.child_mind.components import (
        model_config,
        model_runtime_parameters,
    )
    from reference_projects.kaggle.child_mind.data import (
        PACKED_FEATURES,
        SENSOR_VALUE_SLICE,
        TABULAR_FEATURES,
    )

    runtime = model_runtime_parameters(
        tabular_center=[0.0] * TABULAR_FEATURES,
        tabular_scale=[1.0] * TABULAR_FEATURES,
        sensor_center=[0.0] * (SENSOR_VALUE_SLICE.stop - SENSOR_VALUE_SLICE.start),
        sensor_scale=[1.0] * (SENSOR_VALUE_SLICE.stop - SENSOR_VALUE_SLICE.start),
    )
    model = resolve_component(model_config(hidden=8), **runtime, expected=torch.nn.Module)
    clean = torch.zeros(1, 1, PACKED_FEATURES)
    masked = clean.clone()
    masked[0, 0, 0] = torch.nan
    masked[0, 0, SENSOR_VALUE_SLICE.start] = torch.inf

    assert torch.equal(model(masked), model(clean))

    observed = masked.clone()
    observed[0, 0, TABULAR_FEATURES] = 1
    with pytest.raises(RuntimeError, match="observed values must be finite"):
        model(observed)


@pytest.mark.parametrize(
    ("prediction", "probability", "message"),
    [
        (torch.empty(0, dtype=torch.long), torch.empty(0, 4), "non-empty"),
        (torch.tensor([1.0]), torch.tensor([[0.0, 1.0, 0.0, 0.0]]), "dtype"),
        (
            torch.tensor([1]),
            torch.tensor([[-0.1, 1.1, 0.0, 0.0]]),
            "simplex",
        ),
    ],
)
def test_child_mind_prediction_validation_rejects_invalid_output_values(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    message: str,
) -> None:
    from reference_projects.kaggle.child_mind.components import OUTPUT

    with pytest.raises(ValueError, match=message):
        resolve_component(OUTPUT, expected=MulticlassOutput).validator(
            {"prediction": prediction, "probability": probability}
        )


def test_child_mind_flow_trains_tabular_and_fused_models_with_replayable_evidence(
    child_mind_data: Path,
    tmp_path: Path,
    kaggle_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del kaggle_services
    from lightning import Trainer
    from mlflow import MlflowClient
    from prefect.testing.utilities import prefect_test_harness
    from reference_projects.kaggle.child_mind.flow import child_mind_flow

    observed: list[tuple[type[object], type[object]]] = []
    original = Trainer.fit

    def record(trainer: Trainer, model: object, *args: object, **kwargs: object) -> object:
        observed.append((type(model), type(kwargs["datamodule"])))
        return original(trainer, model, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", record)
    checkout_before = _checkout_status()
    with prefect_test_harness():
        result = child_mind_flow(str(child_mind_data), str(tmp_path / "work"), seed=17)
        replay = child_mind_flow(str(child_mind_data), str(tmp_path / "work"), seed=17)
    # A flow that writes into the consumer checkout changes every later execution identity.
    checkout_after = _checkout_status()
    assert checkout_after == checkout_before, (
        "the checkout changed while the flows ran (a flow wrote into it, or files were edited "
        f"during the test):\nbefore:\n{checkout_before}\nafter:\n{checkout_after}"
    )

    assert result["dataset_digest"] == replay["dataset_digest"]
    assert result["split_digest"] == replay["split_digest"]
    assert result["assignments"] == replay["assignments"]
    assert result["labelled_participants"] == 48
    assert result["dropped_missing_targets"] == 4
    assert result["train_sensor_participants"] == 24
    assert set(result["assignments"]["train"]).isdisjoint(result["assignments"]["validate"])
    assert observed == [(DsioModule, DsioDataModule)] * 4

    for mode in ("tabular", "fused"):
        model = result["models"][mode]
        replay_model = replay["models"][mode]
        assert_replay_run_ids_differ(model, replay_model)
        assert_same_identities(model, replay_model)
        assert_golden_metrics(f"child_mind:{mode}", model["metrics"])
        assert model["metrics"] == replay_model["metrics"]
        assert model["prediction"] == replay_model["prediction"]
        assert model["submission_bytes"] == replay_model["submission_bytes"]
        assert model["prediction_count"] == 4
        assert set(model["metrics"]) == {
            "accuracy",
            "quadratic_weighted_kappa",
            "qwk.sensor_present",
            "qwk.sensor_missing",
            "qwk.sensor_ablated",
            "qwk.sensor_ablation_delta",
        }
        assert all(np.isfinite(value) for value in model["metrics"].values())
        training = MlflowClient().get_run(model["train_run_id"])
        assert 0.0 <= training.data.metrics["train/accuracy"] <= 1.0
        assert 0.0 <= training.data.metrics["val/accuracy"] <= 1.0
        assert model["ablation_sample_ids"] == model["sensor_present_sample_ids"]
        assert model["submission_digest"] == sha256_of_bytes(model["submission_bytes"])
        lines = model["submission_bytes"].decode().splitlines()
        assert lines[0] == "id,sii"
        assert [line.split(",", 1)[0] for line in lines[1:]] == [
            f"test-{index:03d}" for index in range(4)
        ]
        assert all(int(line.rsplit(",", 1)[1]) in range(4) for line in lines[1:])
        evaluation = MlflowClient().get_run(model["evaluation_run_id"])
        for name, value in model["metrics"].items():
            assert evaluation.data.metrics[name] == pytest.approx(value)
        prediction_artifact = MlflowClient().download_artifacts(
            model["evaluation_run_id"],
            "evaluation/predictions.npz",
            str(tmp_path / f"{mode}-evaluation"),
        )
        with np.load(prediction_artifact) as evidence:
            assert (
                evidence["ablation__sensor__sample_id"].tolist()
                == model["sensor_present_sample_ids"]
            )
            assert len(evidence["ablation__sensor__prediction__prediction"]) == len(
                model["ablation_sample_ids"]
            )
        assert_execution_evidence(
            model["train_run_id"],
            tmp_path / f"{mode}-provenance",
            optimizer="torch.optim:AdamW",
            optimizer_parameters={"lr": 0.003, "weight_decay": 0.0001},
            batch_size=16,
            num_workers=0,
        )
        provenance_path = MlflowClient().download_artifacts(
            model["train_run_id"], "provenance.json", str(tmp_path / f"{mode}-fitted")
        )
        provenance = json.loads(Path(provenance_path).read_text())
        fitted = provenance["configuration"]["standardization"]
        assert set(fitted) == {"tabular", "sensor"}
        for name, expected_features in (("tabular", 58), ("sensor", 61)):
            evidence = fitted[name]
            assert evidence["role"] == "train"
            assert evidence["partition"] == 0
            assert evidence["observation"] == "mask"
            assert set(evidence["sample_ids"]) == set(result["assignments"]["train"])
            assert len(evidence["mean"]) == len(evidence["scale"]) == expected_features
            artifact = json.loads(
                Path(
                    MlflowClient().download_artifacts(model["train_run_id"], evidence["artifact"])
                ).read_text()
            )
            assert artifact == {key: value for key, value in evidence.items() if key != "artifact"}
        weights = provenance["configuration"]["weights"]
        assert set(weights) == {"class"}
        class_weights = weights["class"]
        assert class_weights["source"] == "sample_attribute"
        assert class_weights["classes"] == 4
        assert class_weights["sample_ids"] == class_weights["resolved_sample_ids"]
        weight_artifact = json.loads(
            Path(
                MlflowClient().download_artifacts(model["train_run_id"], class_weights["artifact"])
            ).read_text()
        )
        assert weight_artifact == {
            key: value for key, value in class_weights.items() if key != "artifact"
        }
        assert_downstream_evidence(model, tmp_path / f"{mode}-downstream")


def _checkout_status() -> str:
    import subprocess

    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(__file__).parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
