"""Training-role statistics never see held-out samples, and standardization is exact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dsio.data.adapters import SignalExamples, entity_examples
from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.data.views import WindowSpec, build_index
from dsio.experimental.data import (
    fit_class_weights,
    fit_group_weights,
    fit_standardization,
    record_fitted,
)
from dsio.experimental.model import Standardize


def _store(path: Path, overrides: dict[str, float] | None = None) -> SignalStore:
    values = {f"s{index}": float(index) for index in range(8)}
    values.update(overrides or {})
    with SignalStore.builder(path, channels=2) as builder:
        for name, value in values.items():
            builder.add(name, np.array([[value, 5.0], [value + 1, 5.0]]), group=name)
    return SignalStore(path)


def _manifest(store: SignalStore):  # type: ignore[no-untyped-def]
    return generate(
        entity_examples(store), "group_shuffle", name="fit", seed=3, parameters={"test_size": 0.25}
    )


def _assign_training(manifest, sample_ids: list[str]):  # type: ignore[no-untyped-def]
    fold = manifest.fold(0)
    assignments = dict(fold.assignments)
    assignments[manifest.required_roles[0]] = sample_ids
    return manifest.model_copy(
        update={"folds": [fold.model_copy(update={"assignments": assignments})]}
    )


def test_class_weights_fit_only_governed_sample_attributes(tmp_path: Path) -> None:
    path = tmp_path / "class-weights"
    labels = [0, 0, 1, 1, 2, 3, 3, 3]
    with SignalStore.builder(path, channels=1) as builder:
        for index, label in enumerate(labels):
            builder.add(
                f"s{index}",
                np.array([[float(index)]]),
                group=f"g{index}",
                attrs={"target": label},
            )
    store = SignalStore(path)
    training = [f"s{index}" for index in range(7)]
    manifest = _assign_training(_manifest(store), training)

    fitted = fit_class_weights(store, manifest, partition=0, classes=4, attribute="target")

    assert fitted == {
        "classes": 4,
        "counts": [2, 2, 1, 2],
        "weights": [0.875, 0.875, 1.75, 0.875],
        "role": "train",
        "partition": 0,
        "sample_ids": training,
        "resolved_sample_ids": training,
        "source": "sample_attribute",
        "attribute": "target",
    }
    # The held-out label is deliberately changed and cannot affect the governed fit.
    changed_path = tmp_path / "class-weights-changed"
    with SignalStore.builder(changed_path, channels=1) as builder:
        for index, label in enumerate([*labels[:7], 2]):
            builder.add(
                f"s{index}",
                np.array([[float(index)]]),
                group=f"g{index}",
                attrs={"target": label},
            )
    changed = SignalStore(changed_path)
    changed_manifest = _assign_training(_manifest(changed), training)
    assert (
        fit_class_weights(changed, changed_manifest, partition=0, classes=4, attribute="target")[
            "weights"
        ]
        == fitted["weights"]
    )


def test_class_weights_deduplicate_window_entities_and_validate_domain(tmp_path: Path) -> None:
    path = tmp_path / "window-class-weights"
    with SignalStore.builder(path, channels=1) as builder:
        for index in range(4):
            builder.add(
                f"p{index}",
                np.arange(6, dtype=np.float32)[:, None],
                group=f"p{index}",
                attrs={"label": index},
            )
    store = SignalStore(path)
    examples = SignalExamples(store, build_index(store, WindowSpec(length=2, stride=2)))
    manifest = generate(
        examples, "group_shuffle", name="class-windows", seed=3, parameters={"test_size": 0.25}
    )
    training = examples.sample_ids.tolist()
    manifest = _assign_training(manifest, training)

    fitted = fit_class_weights(
        store,
        manifest,
        partition=0,
        classes=4,
        attribute="label",
        examples=examples,
    )

    assert fitted["counts"] == [1, 1, 1, 1]
    assert fitted["weights"] == [1.0] * 4
    assert fitted["resolved_sample_ids"] == [f"p{index}" for index in range(4)]
    assert fitted["source"] == "entity_attribute"

    with pytest.raises(ValueError, match="all 5 classes"):
        fit_class_weights(
            store,
            manifest,
            partition=0,
            classes=5,
            attribute="label",
            examples=examples,
        )


def test_group_weights_fit_training_windows_without_entity_deduplication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "group-weights"
    with SignalStore.builder(path, channels=1) as builder:
        builder.add("a", np.arange(6, dtype=np.float32)[:, None], group="a")
        builder.add("b", np.arange(4, dtype=np.float32)[:, None], group="b")
    store = SignalStore(path)
    examples = SignalExamples(store, build_index(store, WindowSpec(length=2, stride=2)))
    manifest = generate(
        examples, "group_shuffle", name="group-windows", seed=3, parameters={"test_size": 0.5}
    )
    training = examples.sample_ids.tolist()
    manifest = _assign_training(manifest, training)

    fitted = fit_group_weights(store, manifest, partition=0, examples=examples)

    assert fitted["counts"] == {"a": 3, "b": 2}
    assert fitted["group_weights"] == {"a": 5 / 6, "b": 5 / 4}
    assert list(fitted["sample_weights"]) == training
    weights = [fitted["sample_weights"][sample_id] for sample_id in training]
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    for group in ("a", "b"):
        contribution = sum(
            weight
            for weight, actual in zip(weights, examples.groups, strict=True)
            if actual == group
        )
        assert contribution == pytest.approx(2.5)


def test_group_weights_support_store_samples(tmp_path: Path) -> None:
    path = tmp_path / "sample-groups"
    with SignalStore.builder(path, channels=1) as builder:
        for index, group in enumerate(("a", "a", "a", "b")):
            builder.add(f"s{index}", np.zeros((1, 1)), group=group)
    store = SignalStore(path)
    training = ["s0", "s1", "s2", "s3"]
    manifest = _assign_training(_manifest(store), training)

    fitted = fit_group_weights(store, manifest, partition=0)

    assert fitted["source"] == "sample_group"
    assert fitted["counts"] == {"a": 3, "b": 1}
    assert fitted["group_weights"] == {"a": 2 / 3, "b": 2.0}
    assert list(fitted["sample_weights"]) == training


@pytest.mark.parametrize("label", [True, 1.5, -1, 4])
def test_class_weights_reject_malformed_labels(tmp_path: Path, label: object) -> None:
    path = tmp_path / f"bad-label-{label!s}"
    with SignalStore.builder(path, channels=1) as builder:
        for index in range(4):
            builder.add(f"s{index}", np.zeros((1, 1)), group=f"g{index}", attrs={"target": index})
    store = SignalStore(path)
    manifest = _assign_training(_manifest(store), [f"s{index}" for index in range(4)])
    store.entities[0].attrs["target"] = label

    with pytest.raises(ValueError, match="integer class|outside"):
        fit_class_weights(store, manifest, 4, "target", partition=0)


def test_weight_fitters_reject_duplicate_unknown_and_foreign_assignments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weight-boundaries"
    with SignalStore.builder(path, channels=1) as builder:
        for index in range(4):
            builder.add(
                f"s{index}",
                np.zeros((2, 1)),
                group=f"g{index}",
                attrs={"target": index},
            )
    store = SignalStore(path)
    manifest = _manifest(store)
    for invalid in (["s0", "s0"], ["missing"]):
        changed = _assign_training(manifest, invalid)
        with pytest.raises(ValueError, match="repeats|unknown"):
            fit_class_weights(store, changed, 4, "target", partition=0)
        with pytest.raises(ValueError, match="repeats|unknown"):
            fit_group_weights(store, changed, partition=0)

    examples = SignalExamples(store, build_index(store, WindowSpec(length=1, stride=1)))
    window_manifest = generate(
        examples, "group_shuffle", name="foreign", seed=3, parameters={"test_size": 0.25}
    )
    foreign_path = tmp_path / "foreign-name"
    with SignalStore.builder(foreign_path, channels=1) as builder:
        for index in range(4):
            builder.add(
                f"s{index}",
                np.zeros((2, 1)),
                group=f"g{index}",
                attrs={"target": index},
            )
    foreign = SignalStore(foreign_path)
    foreign_examples = SignalExamples(foreign, build_index(foreign, WindowSpec(length=1, stride=1)))
    with pytest.raises(ValueError, match="same store name and content identity"):
        fit_class_weights(
            store,
            window_manifest,
            4,
            "target",
            partition=0,
            examples=foreign_examples,
        )
    with pytest.raises(ValueError, match="same store name and content identity"):
        fit_group_weights(store, window_manifest, partition=0, examples=foreign_examples)


def test_statistics_use_only_the_training_role(tmp_path: Path) -> None:
    store = _store(tmp_path / "base")
    manifest = _manifest(store)
    held_out = list(manifest.fold(0).assignments["test"])

    fitted = fit_standardization(store, manifest, partition=0)
    rows = np.concatenate(
        [store.read_sample(sample_id)["data"] for sample_id in fitted["sample_ids"]]
    ).astype(np.float64)

    assert set(fitted["sample_ids"]) == set(manifest.fold(0).assignments["train"])
    assert fitted["mean"] == rows.mean(axis=0).tolist()
    assert fitted["scale"] == [rows.std(axis=0)[0], 1.0]  # zero-variance feature maps to 1

    tampered = _store(tmp_path / "tampered", {sample_id: 1000.0 for sample_id in held_out})
    tampered_manifest = _manifest(tampered)
    assert tampered_manifest.fold(0).assignments == manifest.fold(0).assignments
    refitted = fit_standardization(tampered, tampered_manifest, partition=0)
    assert (refitted["mean"], refitted["scale"]) == (fitted["mean"], fitted["scale"])


def test_fitting_refuses_a_held_out_role_and_a_different_store(tmp_path: Path) -> None:
    store = _store(tmp_path / "original" / "store")
    manifest = _manifest(store)

    with pytest.raises(TypeError, match="unexpected keyword argument 'role'"):
        fit_standardization(store, manifest, partition=0, role="test")  # type: ignore[call-arg]

    other = _store(tmp_path / "different" / "store", {"s0": 99.0})
    with pytest.raises(ValueError, match="dataset digest"):
        fit_standardization(other, manifest, partition=0)


def test_fitting_uses_the_manifests_declared_training_role(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    manifest = generate(
        entity_examples(store),
        "group_shuffle",
        name="custom-roles",
        seed=3,
        roles=("fit", "calibrate"),
        parameters={"test_size": 0.25},
    )

    fitted = fit_standardization(store, manifest, partition=0)

    assert fitted["role"] == "fit"
    assert fitted["sample_ids"] == manifest.fold(0).assignments["fit"]


@pytest.mark.parametrize("missing", [False, True], ids=["empty", "missing"])
def test_fitting_rejects_an_invalid_training_assignment_before_reading_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: bool,
) -> None:
    store = _store(tmp_path / "store")
    manifest = _manifest(store)
    fold = manifest.fold(0)
    assignments = dict(fold.assignments)
    if missing:
        assignments.pop("train")
    else:
        assignments["train"] = []
    invalid_fold = fold.model_copy(update={"assignments": assignments})
    invalid_manifest = manifest.model_copy(update={"folds": [invalid_fold]})

    def forbidden_read(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("sample data was read before the manifest boundary was validated")

    monkeypatch.setattr(SignalStore, "read_sample", forbidden_read)
    with pytest.raises(ValueError, match="missing declared training role|assigns no samples"):
        fit_standardization(store, invalid_manifest, partition=0)


def test_fitting_requires_the_manifest_to_declare_a_training_role_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "store")
    manifest = _manifest(store).model_copy(update={"required_roles": ()})

    def forbidden_read(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("sample data was read before the manifest boundary was validated")

    monkeypatch.setattr(SignalStore, "read_sample", forbidden_read)
    with pytest.raises(ValueError, match="declares no training role"):
        fit_standardization(store, manifest, partition=0)


def test_observed_statistics_ignore_missing_values_and_refuse_empty_features(
    tmp_path: Path,
) -> None:
    with SignalStore.builder(tmp_path / "nan", channels=2) as builder:
        for index in range(4):
            builder.add(f"s{index}", np.array([[float(index), np.nan]]), group=f"g{index}")
    store = SignalStore(tmp_path / "nan")
    manifest = _manifest(store)

    with pytest.raises(ValueError, match=r"features \[1\] have no observed value"):
        fit_standardization(store, manifest, partition=0, observed=True)
    assert np.isnan(fit_standardization(store, manifest, partition=0)["mean"][1])


def test_explicit_observation_columns_select_before_arithmetic(tmp_path: Path) -> None:
    path = tmp_path / "masked"
    with SignalStore.builder(path, channels=4) as builder:
        for index in range(8):
            builder.add(
                f"s{index}",
                np.array([[float(index), np.nan, 1.0, 0.0]]),
                group=f"g{index}",
            )
    store = SignalStore(path)
    manifest = _manifest(store)

    with pytest.raises(ValueError, match=r"features \['empty'\] have no observed value"):
        fit_standardization(
            store,
            manifest,
            partition=0,
            value_columns=[0, 1],
            observed_columns=[2, 3],
            feature_names=["value", "empty"],
        )

    fitted = fit_standardization(
        store,
        manifest,
        partition=0,
        value_columns=[0],
        observed_columns=[2],
        feature_names=["value"],
        minimum_scale=1e-6,
        reduction="sum_squares",
    )
    training = np.array([float(name[1:]) for name in fitted["sample_ids"]])
    assert fitted["mean"] == [training.mean()]
    assert fitted["scale"] == [training.std()]
    assert fitted["observation"] == "mask"
    assert fitted["resolved_sample_ids"] == fitted["sample_ids"]


def test_nondefault_policy_is_complete_and_reduction_is_honored_without_masks(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "policy")
    manifest = _manifest(store)

    fitted = fit_standardization(
        store,
        manifest,
        partition=0,
        observed=True,
        feature_names=["varying", "constant"],
        minimum_scale=1e-6,
        reduction="sum_squares",
    )
    rows = np.concatenate(
        [store.read_sample(sample_id)["data"] for sample_id in fitted["sample_ids"]]
    ).astype(np.float64)
    expected_mean = rows.sum(axis=0) / len(rows)
    expected_scale = np.sqrt(
        np.maximum(np.square(rows).sum(axis=0) / len(rows) - expected_mean**2, 0.0)
    )
    expected_scale[expected_scale < 1e-6] = 1.0

    assert fitted["mean"] == expected_mean.tolist()
    assert fitted["scale"] == expected_scale.tolist()
    assert fitted["source"] == "data"
    assert fitted["observation"] == "nan"
    assert fitted["resolved_sample_ids"] == fitted["sample_ids"]
    assert fitted["feature_names"] == ["varying", "constant"]
    assert fitted["minimum_scale"] == 1e-6
    assert fitted["reduction"] == "sum_squares"


def test_source_configuration_is_rejected_before_payload_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "validation")
    manifest = _manifest(store)

    def forbidden_read(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("payload read before source configuration validation")

    monkeypatch.setattr(SignalStore, "read_sample", forbidden_read)
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        fit_standardization(store, manifest, partition=0, value_columns=[2])


def test_fitting_rejects_zero_width_sources_and_reduction_overflow(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty-attribute"
    with SignalStore.builder(empty_path, channels=1) as builder:
        for index in range(4):
            builder.add(
                f"p{index}",
                np.arange(4, dtype=np.float32)[:, None],
                group=f"p{index}",
                attrs={"empty": []},
            )
    empty_store = SignalStore(empty_path)
    empty_examples = SignalExamples(
        empty_store, build_index(empty_store, WindowSpec(length=2, stride=2))
    )
    empty_manifest = generate(
        empty_examples,
        "group_shuffle",
        name="empty",
        seed=3,
        parameters={"test_size": 0.25},
    )
    with pytest.raises(ValueError, match="at least one feature"):
        fit_standardization(
            empty_store,
            empty_manifest,
            partition=0,
            examples=empty_examples,
            attribute="empty",
        )

    overflow_path = tmp_path / "overflow"
    with SignalStore.builder(overflow_path, channels=1, dtype="float64") as builder:
        for index in range(4):
            builder.add(f"s{index}", np.array([[1e308]]), group=f"g{index}")
    overflow_store = SignalStore(overflow_path)
    with pytest.raises(ValueError, match="source values overflowed"):
        fit_standardization(
            overflow_store,
            _manifest(overflow_store),
            partition=0,
            reduction="sum_squares",
        )


def test_window_assignments_fit_unique_entity_attributes(tmp_path: Path) -> None:
    path = tmp_path / "window-attrs"
    with SignalStore.builder(path, channels=1) as builder:
        for index in range(4):
            builder.add(
                f"p{index}",
                np.arange(6, dtype=np.float32)[:, None],
                group=f"p{index}",
                attrs={"packed": [float(index), float(index + 10), 1.0, 1.0]},
            )
    store = SignalStore(path)
    examples = SignalExamples(store, build_index(store, WindowSpec(length=2, stride=2)))
    manifest = generate(
        examples, "group_shuffle", name="windows", seed=3, parameters={"test_size": 0.25}
    )

    fitted = fit_standardization(
        store,
        manifest,
        partition=0,
        examples=examples,
        attribute="packed",
        value_columns=[0, 1],
        observed_columns=[2, 3],
        minimum_scale=1e-6,
    )

    training = manifest.fold(0).assignments[manifest.required_roles[0]]
    positions = {sample_id: position for position, sample_id in enumerate(examples.sample_ids)}
    expected_ids = list(
        dict.fromkeys(
            examples.index.entity_names[int(examples.index.entity_codes[positions[sample_id]])]
            for sample_id in training
        )
    )
    expected = np.asarray([store.entity(name).attrs["packed"][:2] for name in expected_ids])
    assert fitted["sample_ids"] == training
    assert fitted["resolved_sample_ids"] == expected_ids
    assert fitted["mean"] == expected.mean(axis=0).tolist()
    assert fitted["scale"] == expected.std(axis=0).tolist()


def test_window_attribute_fitting_rejects_foreign_unknown_and_missing_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "governed"
    with SignalStore.builder(path, channels=1) as builder:
        for index in range(4):
            builder.add(
                f"p{index}",
                np.arange(4, dtype=np.float32)[:, None],
                group=f"p{index}",
                attrs={"packed": [float(index), 1.0]},
            )
    store = SignalStore(path)
    examples = SignalExamples(store, build_index(store, WindowSpec(length=2, stride=2)))
    manifest = generate(
        examples, "group_shuffle", name="governed", seed=3, parameters={"test_size": 0.25}
    )

    foreign_path = tmp_path / "foreign"
    with SignalStore.builder(foreign_path, channels=1) as builder:
        builder.add("foreign", np.ones((4, 1), dtype=np.float32), group="foreign")
    foreign_store = SignalStore(foreign_path)
    foreign_examples = SignalExamples(
        foreign_store, build_index(foreign_store, WindowSpec(length=2, stride=2))
    )
    with pytest.raises(ValueError, match="same store name and content identity"):
        fit_standardization(
            store,
            manifest,
            partition=0,
            examples=foreign_examples,
            attribute="packed",
        )

    fold = manifest.fold(0)
    assignments = dict(fold.assignments)
    assignments["train"] = ["missing-window"]
    unknown_manifest = manifest.model_copy(
        update={"folds": [fold.model_copy(update={"assignments": assignments})]}
    )
    with pytest.raises(ValueError, match="unknown window identity"):
        fit_standardization(
            store,
            unknown_manifest,
            partition=0,
            examples=examples,
            attribute="packed",
        )
    with pytest.raises(ValueError, match="has no attribute 'missing'"):
        fit_standardization(
            store,
            manifest,
            partition=0,
            examples=examples,
            attribute="missing",
        )


def test_observed_fitting_is_isolated_from_held_out_values(tmp_path: Path) -> None:
    def build(path: Path, overrides: dict[str, float] | None = None) -> SignalStore:
        changed = overrides or {}
        with SignalStore.builder(path, channels=2) as builder:
            for index in range(8):
                name = f"s{index}"
                builder.add(
                    name,
                    np.array([[changed.get(name, float(index)), 1.0]]),
                    group=name,
                )
        return SignalStore(path)

    store = build(tmp_path / "observed-base")
    manifest = _manifest(store)
    held_out = manifest.fold(0).assignments["test"]
    fitted = fit_standardization(
        store,
        manifest,
        partition=0,
        value_columns=[0],
        observed_columns=[1],
    )

    tampered = build(tmp_path / "observed-tampered", {name: 1e6 for name in held_out})
    tampered_manifest = _manifest(tampered)
    refitted = fit_standardization(
        tampered,
        tampered_manifest,
        partition=0,
        value_columns=[0],
        observed_columns=[1],
    )
    assert tampered_manifest.fold(0).assignments == manifest.fold(0).assignments
    assert (refitted["mean"], refitted["scale"]) == (fitted["mean"], fitted["scale"])


def test_observation_columns_reject_nonbinary_values(tmp_path: Path) -> None:
    path = tmp_path / "invalid-mask"
    with SignalStore.builder(path, channels=2) as builder:
        for index in range(4):
            builder.add(f"s{index}", np.array([[float(index), 0.5]]), group=f"g{index}")
    store = SignalStore(path)
    with pytest.raises(ValueError, match="finite zero or one"):
        fit_standardization(
            store,
            _manifest(store),
            partition=0,
            value_columns=[0],
            observed_columns=[1],
        )


def test_standardize_matches_the_consumer_arithmetic_bit_for_bit() -> None:
    mean, scale = [0.5, -2.0, 3.25], [1.5, 1.0, 0.75]
    x = torch.randn(4, 1, 3)
    expected = (x.float() - torch.tensor(mean).reshape(1, 1, -1)) / torch.tensor(scale).reshape(
        1, 1, -1
    )
    assert torch.equal(Standardize(mean, scale)(x), expected)

    pixels = torch.randint(0, 256, (2, 28, 28), dtype=torch.uint8)
    assert torch.equal(Standardize([0.0], [255.0])(pixels), pixels.float() / 255.0)


def test_standardize_checks_features_axis_and_statistics() -> None:
    channel_first = Standardize([1.0, 2.0], [1.0, 2.0], axis=1)
    assert channel_first(torch.ones(1, 2, 3)).shape == (1, 2, 3)
    with pytest.raises(ValueError, match="expected 2 features on axis 1"):
        channel_first(torch.ones(1, 3, 2))
    with pytest.raises(ValueError, match="same length"):
        Standardize([0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-zero"):
        Standardize([0.0], [0.0])


def test_fitted_values_are_logged_as_run_relative_evidence() -> None:
    from mlflow import MlflowClient

    client = MlflowClient()
    run = client.create_run(client.create_experiment("fitting-evidence"))
    path = record_fitted(run.info.run_id, "standardization", {"mean": [1.0], "scale": [2.0]})

    assert path == "fitted/standardization.json"
    downloaded = Path(client.download_artifacts(run.info.run_id, path))
    assert downloaded.read_text(encoding="utf-8").count('"mean"') == 1
    with pytest.raises(ValueError, match="invalid fitted artifact name"):
        record_fitted(run.info.run_id, "../escape", {})
    with pytest.raises(ValueError, match="finite canonical JSON"):
        record_fitted(run.info.run_id, "invalid", {"mean": [float("nan")]})


def test_standardize_casts_to_its_statistics_dtype() -> None:
    stage = Standardize([1.0], [2.0])
    assert stage(torch.ones(2, 1, dtype=torch.float64)).dtype == torch.float32
    assert stage.double()(torch.ones(2, 1)).dtype == torch.float64
