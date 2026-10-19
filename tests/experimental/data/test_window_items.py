"""WindowItems maps governed windows into configured training fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dsio.config.components import ComponentError, resolve_component
from dsio.data.adapters import SignalExamples
from dsio.data.loading import LoadingError, build_loader
from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex, WindowSpec, build_index
from dsio.experimental.data import WindowItems


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[SignalStore, SignalExamples]:
    path = tmp_path / "windows"
    with SignalStore.builder(path, channels=2) as builder:
        builder.add(
            "a",
            np.arange(12, dtype=np.float32).reshape(6, 2),
            group="g-a",
            attrs={"label": 1, "packed": [10.0, 11.0], "present": True},
        )
        builder.add(
            "b",
            np.arange(8, dtype=np.float32).reshape(4, 2) + 20,
            group="g-b",
            attrs={"label": 0, "packed": [20.0, 21.0], "present": False},
        )
    store = SignalStore(path)
    index = build_index(store, WindowSpec(length=2, stride=2))
    index.labels = np.arange(len(index), dtype=np.int64)
    index.metrics["quality"] = np.asarray([1, 0, 1, 0, 1], dtype=np.int64)
    return store, SignalExamples(store, index)


def _factory(sample_weights: dict[str, float] | None = None) -> WindowItems:
    return WindowItems(
        x={
            "parts": [
                {"from": "entity_attribute", "attribute": "packed", "shape": [-1]},
                {"from": "window", "shape": [-1]},
                {
                    "from": "entity_attribute",
                    "attribute": "present",
                    "shape": [1],
                },
            ]
        },
        y={"from": "entity_attribute", "attribute": "label", "dtype": "int64"},
        mask={"from": "index_metric", "metric": "quality", "dtype": "bool"},
        sample_weight={"from": "fitted"},
        fitted_sample_weights=sample_weights,
        group=True,
    )


def test_maps_composite_fields_group_and_fitted_weights(
    corpus: tuple[SignalStore, SignalExamples],
) -> None:
    store, examples = corpus
    ids = [examples.index.sample_id(position) for position in (3, 0, 1)]
    dataset = _factory(dict(zip(ids, [1.5, 0.75, 0.75], strict=True)))(store, examples, ids)

    items = [dataset[position] for position in range(len(dataset))]

    assert [item["sample_id"] for item in items] == ids
    assert [item["group"] for item in items] == ["g-b", "g-a", "g-a"]
    expected_fields = {"sample_id", "x", "y", "mask", "sample_weight", "group"}
    assert all(set(item) == expected_fields for item in items)
    assert items[0]["x"].dtype == torch.float32
    assert items[0]["x"].tolist() == [20.0, 21.0, 20.0, 21.0, 22.0, 23.0, 0.0]
    assert items[0]["y"].dtype == torch.int64 and items[0]["y"].shape == ()
    assert items[0]["mask"].dtype == torch.bool
    assert items[0]["mask"].item() is bool(examples.index.metrics["quality"][3])
    assert [float(item["sample_weight"]) for item in items] == pytest.approx([1.5, 0.75, 0.75])
    assert sum(float(item["sample_weight"]) for item in items) / len(items) == pytest.approx(1.0)


def test_fitted_weights_are_identity_aligned_and_disjoint_roles_are_neutral(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared-group"
    with SignalStore.builder(path, channels=1) as builder:
        builder.add("a", np.arange(6, dtype=np.float32)[:, None], group="shared")
        builder.add("b", np.arange(4, dtype=np.float32)[:, None], group="shared")
    store = SignalStore(path)
    examples = SignalExamples(store, build_index(store, WindowSpec(length=2, stride=2)))
    dataset = WindowItems(
        x={"from": "window"},
        sample_weight={"from": "fitted"},
        group=True,
        fitted_sample_weights={sample_id: 2.0 for sample_id in examples.sample_ids[:2]},
    )(store, examples, examples.sample_ids[:2].tolist())

    items = [dataset[position] for position in range(len(dataset))]

    assert {item["group"] for item in items} == {"shared"}
    assert [float(item["sample_weight"]) for item in items] == pytest.approx([2.0, 2.0])

    held_out = WindowItems(
        x={"from": "window"},
        sample_weight={"from": "fitted"},
        fitted_sample_weights={sample_id: 2.0 for sample_id in examples.sample_ids[:2]},
    )(store, examples, examples.sample_ids[2:].tolist())
    assert [float(held_out[index]["sample_weight"]) for index in range(len(held_out))] == [
        1.0,
        1.0,
        1.0,
    ]

    with pytest.raises(LoadingError, match="partial fitted sample-weight coverage"):
        WindowItems(
            x={"from": "window"},
            sample_weight={"from": "fitted"},
            fitted_sample_weights={examples.sample_ids[0]: 1.0},
        )(store, examples, examples.sample_ids[:2].tolist())


def test_index_label_and_single_window_layout_are_supported(
    corpus: tuple[SignalStore, SignalExamples],
) -> None:
    store, examples = corpus
    sample_id = examples.index.sample_id(0)
    dataset = WindowItems(
        x={"from": "window", "layout": "channel_first"},
        y={"from": "index_label", "dtype": "int64"},
    )(store, examples, [sample_id])

    item = dataset[0]
    assert item["x"].shape == (2, 2)
    assert item["x"].tolist() == [[0.0, 2.0], [1.0, 3.0]]
    assert item["y"].item() == 0


def test_bad_inputs_name_the_field_and_window(
    corpus: tuple[SignalStore, SignalExamples], tmp_path: Path
) -> None:
    store, examples = corpus
    sample_id = examples.index.sample_id(0)
    with pytest.raises(ValueError, match="field 'x'.*unsupported source"):
        WindowItems(x={"from": "callback"})
    with pytest.raises(LoadingError, match="field 'y'.*missing.*sample"):
        WindowItems(x={"from": "window"}, y={"from": "index_metric", "metric": "missing"})(
            store, examples, [sample_id]
        )[0]
    with pytest.raises(LoadingError, match="unknown window sample identity 'missing'"):
        WindowItems(x={"from": "window"})(store, examples, ["missing"])
    with pytest.raises(LoadingError, match="duplicate assigned window sample identity"):
        WindowItems(x={"from": "window"})(store, examples, [sample_id, sample_id])

    other_path = tmp_path / "other"
    with SignalStore.builder(other_path, channels=2) as builder:
        builder.add("z", np.zeros((2, 2), dtype=np.float32), group="z")
    with pytest.raises(LoadingError, match="SignalExamples.*same store"):
        WindowItems(x={"from": "window"})(SignalStore(other_path), examples, [sample_id])


def test_index_metadata_and_window_ownership_are_validated(
    corpus: tuple[SignalStore, SignalExamples],
) -> None:
    store, examples = corpus
    index = examples.index
    short_labels = WindowIndex(
        starts=index.starts,
        entity_codes=index.entity_codes,
        entity_names=index.entity_names,
        entity_groups=index.entity_groups,
        spec=index.spec,
        store_name=index.store_name,
        store_digest=index.store_digest,
        labels=np.zeros(len(index) - 1, dtype=np.int64),
        metrics=index.metrics,
    )
    with pytest.raises(LoadingError, match="index labels.*values"):
        WindowItems(x={"from": "window"})(
            store, SignalExamples(store, short_labels), [short_labels.sample_id(0)]
        )

    short_metric = WindowIndex(
        starts=index.starts,
        entity_codes=index.entity_codes,
        entity_names=index.entity_names,
        entity_groups=index.entity_groups,
        spec=index.spec,
        store_name=index.store_name,
        store_digest=index.store_digest,
        labels=index.labels,
        metrics={"short": np.zeros(len(index) - 1, dtype=np.float32)},
    )
    with pytest.raises(LoadingError, match="index metric 'short'.*values"):
        WindowItems(x={"from": "window"})(
            store, SignalExamples(store, short_metric), [short_metric.sample_id(0)]
        )

    wrong_codes = index.entity_codes.copy()
    wrong_codes[0] = 1
    wrong_owner = WindowIndex(
        starts=index.starts,
        entity_codes=wrong_codes,
        entity_names=index.entity_names,
        entity_groups=index.entity_groups,
        spec=index.spec,
        store_name=index.store_name,
        store_digest=index.store_digest,
        labels=index.labels,
        metrics=index.metrics,
    )
    wrong_id = wrong_owner.sample_id(0)
    with pytest.raises(LoadingError, match=f"window {wrong_id!r}.*is not owned"):
        WindowItems(x={"from": "window"})(store, SignalExamples(store, wrong_owner), [wrong_id])


def test_mapping_transforms_failures_and_index_namespaces(
    corpus: tuple[SignalStore, SignalExamples],
) -> None:
    store, examples = corpus
    sample_id = examples.index.sample_id(0)
    store.entities[0].attrs["__index_label__"] = 99
    item = WindowItems(
        x={
            "parts": [
                {
                    "from": "window",
                    "columns": [1, 2],
                    "offset": 1.0,
                    "divide": 2.0,
                    "finite": True,
                    "shape": [-1],
                },
                {
                    "from": "entity_attribute",
                    "attribute": "__index_label__",
                    "shape": [1],
                },
                {"from": "index_label", "shape": [1]},
            ]
        }
    )(store, examples, [sample_id])[0]
    assert item["x"].tolist() == [1.0, 2.0, 99.0, 0.0]

    with pytest.raises(LoadingError, match="field 'x' cannot reshape.*sample"):
        WindowItems(x={"from": "window", "shape": [3]})(store, examples, [sample_id])[0]
    examples.index.metrics["not_finite"] = np.full(len(examples.index), np.nan)
    with pytest.raises(LoadingError, match="field 'y'.*sample.*finite"):
        WindowItems(
            x={"from": "window"},
            y={"from": "index_metric", "metric": "not_finite", "finite": True},
        )(store, examples, [sample_id])[0]
    with pytest.raises(ValueError, match="expected.*fitted"):
        WindowItems(x={"from": "window"}, sample_weight={"from": "other"})
    with pytest.raises(ValueError, match="finite and positive"):
        WindowItems(
            x={"from": "window"},
            sample_weight={"from": "fitted"},
            fitted_sample_weights={sample_id: float("nan")},
        )
    with pytest.raises(ValueError, match="float32"):
        WindowItems(
            x={"from": "window"},
            sample_weight={"from": "fitted"},
            fitted_sample_weights={sample_id: 1e300},
        )
    with pytest.raises(ValueError, match="requires runtime fitted_sample_weights"):
        WindowItems(x={"from": "window"}, sample_weight={"from": "fitted"})


def test_configured_fitted_weights_cannot_override_runtime_injection() -> None:
    configured = {
        "reference": "dsio.experimental.data.windows:WindowItems",
        "parameters": {
            "x": {"from": "window"},
            "sample_weight": {"from": "fitted"},
            "fitted_sample_weights": {"sample": 1.0},
        },
    }
    with pytest.raises(ComponentError, match="fitted_sample_weights.*collide with runtime"):
        resolve_component(configured, fitted_sample_weights={"sample": 2.0})


def test_first_window_read_verifies_the_owning_payload(
    corpus: tuple[SignalStore, SignalExamples],
) -> None:
    store, examples = corpus
    dataset = WindowItems(x={"from": "window"})(store, examples, [examples.index.sample_id(0)])
    payload = next(Path(store.path).rglob("*.bin"))
    raw = bytearray(payload.read_bytes())
    raw[0] ^= 0xFF
    payload.write_bytes(bytes(raw))

    with pytest.raises(LoadingError, match="digest"):
        dataset[0]


def test_an_entity_is_verified_once_per_dataset_process(
    corpus: tuple[SignalStore, SignalExamples], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, examples = corpus
    dataset = WindowItems(x={"from": "window"})(
        store, examples, [examples.index.sample_id(0), examples.index.sample_id(1)]
    )
    original = store.read_sample
    calls: list[str] = []

    def record(entity_id: str) -> object:
        calls.append(entity_id)
        return original(entity_id)

    monkeypatch.setattr(store, "read_sample", record)
    _ = dataset[0]
    _ = dataset[1]

    assert calls == ["a"]


def test_preverified_dataset_reverifies_in_spawn_worker(
    corpus: tuple[SignalStore, SignalExamples],
) -> None:
    store, examples = corpus
    dataset = WindowItems(x={"from": "window"})(store, examples, [examples.index.sample_id(0)])
    _ = dataset[0]
    payload = next(Path(store.path).rglob("*.bin"))
    raw = bytearray(payload.read_bytes())
    raw[0] ^= 0xFF
    payload.write_bytes(bytes(raw))

    with pytest.raises(LoadingError, match="digest"):
        next(iter(build_loader(dataset, batch_size=1, num_workers=1)))


def test_worker_count_preserves_items_with_relative_store_path(
    corpus: tuple[SignalStore, SignalExamples], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, examples = corpus
    monkeypatch.chdir(store.path.parent)
    store = SignalStore(Path(store.path.name))
    examples = SignalExamples(store, examples.index)
    ids = examples.sample_ids.tolist()
    weights = {sample_id: float(index + 1) for index, sample_id in enumerate(ids)}

    def load(
        workers: int,
    ) -> tuple[list[str], torch.Tensor, list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        batches = list(
            build_loader(
                _factory(weights)(store, examples, ids),
                batch_size=2,
                shuffle=True,
                num_workers=workers,
                seed=17,
            )
        )
        return (
            [value for batch in batches for value in batch["sample_id"]],
            torch.cat([batch["x"] for batch in batches]),
            [value for batch in batches for value in batch["group"]],
            torch.cat([batch["y"] for batch in batches]),
            torch.cat([batch["mask"] for batch in batches]),
            torch.cat([batch["sample_weight"] for batch in batches]),
        )

    zero, two = load(0), load(2)
    assert zero[0] == two[0]
    assert torch.equal(zero[1], two[1])
    assert zero[2] == two[2]
    assert all(torch.equal(left, right) for left, right in zip(zero[3:], two[3:], strict=True))
