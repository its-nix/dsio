"""A split over a store's entities reaches its window views without loosening any gate."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dsio.data.adapters import SignalExamples, entity_examples
from dsio.data.splits import generate, validate
from dsio.data.splits.models import SplitError, SplitFile, SplitFold
from dsio.data.splits.resolve import assert_no_row_overlap, resolve_masks
from dsio.data.splits.temporal import TemporalBounds, TimeSpan
from dsio.data.splits.windows import resolve_window_masks, window_evidence
from dsio.data.store import SignalStore
from dsio.data.views import (
    ViewError,
    WindowIndex,
    WindowSpec,
    assert_windows_within_entities,
    build_index,
)


def _store(path: Path, *, entity_ids: Sequence[str] | None = None) -> SignalStore:
    """Six people, two recordings each, so every entity holds several windows."""
    names = entity_ids or [f"p{person}_s{session}" for person in range(6) for session in range(2)]
    with SignalStore.builder(path, channels=1) as builder:
        for position, name in enumerate(names):
            builder.add(
                name,
                np.full((8, 1), position, dtype=np.float32),
                group=f"p{position // 2}",
                attrs={"session": position % 2},
            )
    return SignalStore(path)


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    return _store(tmp_path / "cohort")


def _view(store: SignalStore) -> SignalExamples:
    return SignalExamples(store, build_index(store, WindowSpec(length=4, stride=2)))


def _kfold(store: SignalStore) -> SplitFile:
    return generate(
        entity_examples(store), "group_kfold", name="people", seed=3, parameters={"n_splits": 3}
    )


def _session(view: SignalExamples, session: int) -> SignalExamples:
    return view.subset(view.attribute("session") == session)


def _nested(store: SignalStore) -> tuple[SplitFile, SplitFold]:
    """A validation split over the training entities of an outer fold."""
    outer = _kfold(store)
    fold = outer.fold(0)
    entities = entity_examples(store)
    inside = entities.subset(np.isin(entities.sample_ids, fold.assignments["train"]))
    inner = generate(
        inside,
        "group_shuffle",
        name="inner",
        seed=5,
        roles=("fit", "val"),
        parameters={"test_size": 0.25},
    )
    return inner, fold


def test_masks_partition_a_whole_view(store: SignalStore) -> None:
    view = _view(store)
    split = _kfold(store)

    for fold in split.folds:
        masks = resolve_window_masks(view, split, fold)

        assert set(masks) == {"train", "test"}
        assert np.array_equal(masks["train"] ^ masks["test"], np.ones(len(view), dtype=bool))
        assert_no_row_overlap({role: view.subset(mask) for role, mask in masks.items()})
        groups = {role: set(view.groups[mask].tolist()) for role, mask in masks.items()}
        assert groups == {role: set(fold.parts[role]) for role in masks}


def test_one_split_replays_across_filtered_views(store: SignalStore) -> None:
    view = _view(store)
    split = _kfold(store)
    fold = split.fold(1)

    first = resolve_window_masks(_session(view, 0), split, fold)
    second = resolve_window_masks(_session(view, 1), split, fold)

    for role in fold.parts:
        for session, masks in ((0, first), (1, second)):
            windows = _session(view, session).subset(masks[role])
            assert set(windows.groups.tolist()) == set(fold.parts[role])
            assert set(windows.attribute("session").tolist()) == {session}


def test_a_nested_split_resolves_on_a_view_of_its_population(store: SignalStore) -> None:
    inner, outer = _nested(store)
    view = _view(store)
    inside = view.subset(np.isin(view.index.entity_ids, outer.assignments["train"]))

    masks = resolve_window_masks(inside, inner, inner.fold(0))

    assert len(inside) > len(outer.assignments["train"])
    assert np.array_equal(masks["fit"] ^ masks["val"], np.ones(len(inside), dtype=bool))


def test_a_nested_split_refuses_windows_outside_its_population(store: SignalStore) -> None:
    inner, _ = _nested(store)

    with pytest.raises(SplitError, match="no owner|exactly one owner"):
        resolve_window_masks(_view(store), inner, inner.fold(0))


def test_a_fold_that_differs_from_the_manifest_is_refused(store: SignalStore) -> None:
    split = _kfold(store)
    fold = split.fold(0).model_copy(update={"counts": {"train": 1, "test": 1}})

    with pytest.raises(SplitError, match="does not match"):
        resolve_window_masks(_view(store), split, fold)


def test_a_temporal_algorithm_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "long"
    with SignalStore.builder(path, channels=1) as builder:
        builder.add("only", np.zeros((400, 1), dtype=np.float32), group="p0")
    store = SignalStore(path)
    view = SignalExamples(store, build_index(store, WindowSpec(length=4, stride=4)))
    split = generate(
        view,
        "purged_walk_forward",
        name="time",
        seed=0,
        parameters={"n_splits": 2, "test_fraction": 0.2},
    )

    with pytest.raises(SplitError, match="time"):
        resolve_window_masks(view, split, split.folds[0])


def test_a_grouped_manifest_with_temporal_bounds_is_refused(store: SignalStore) -> None:
    split = _kfold(store)
    bounds = TemporalBounds(
        time_unit="row",
        label_horizon=0.0,
        embargo=0.0,
        spans={"train": [TimeSpan(start=0.0, end=4.0)]},
    )
    folds = [
        fold.model_copy(update={"temporal": bounds}) if fold.index == 0 else fold
        for fold in split.folds
    ]
    bounded = split.model_copy(update={"folds": folds})

    with pytest.raises(SplitError, match="time"):
        resolve_window_masks(_view(store), bounded, bounded.fold(1))


def test_partial_coverage_is_refused(store: SignalStore) -> None:
    split = _kfold(store).model_copy(update={"coverage": "partial"})

    with pytest.raises(SplitError, match="time"):
        resolve_window_masks(_view(store), split, split.fold(0))


def test_a_window_level_split_is_refused(store: SignalStore) -> None:
    view = _view(store)
    split = generate(view, "group_kfold", name="windows", seed=3, parameters={"n_splits": 3})

    with pytest.raises(SplitError):
        resolve_window_masks(view, split, split.fold(0))


def test_a_split_of_another_store_is_refused(store: SignalStore, tmp_path: Path) -> None:
    other = _store(tmp_path / "other", entity_ids=[f"o{index}" for index in range(12)])

    with pytest.raises(SplitError):
        resolve_window_masks(_view(store), _kfold(other), _kfold(other).fold(0))


def _rebuilt(index: WindowIndex, **changes: Any) -> WindowIndex:
    fields: dict[str, Any] = {
        "starts": index.starts,
        "entity_codes": index.entity_codes,
        "entity_names": index.entity_names,
        "entity_groups": index.entity_groups,
        "spec": index.spec,
        "store_name": index.store_name,
        "store_digest": index.store_digest,
        "labels": index.labels,
        "metrics": index.metrics,
    }
    return WindowIndex(**(fields | changes))


def test_a_window_outside_its_entity_is_refused(store: SignalStore) -> None:
    view = _view(store)
    split = _kfold(store)
    misowned = view.index.entity_codes.copy()
    misowned[0] = 2

    with pytest.raises(ViewError, match="outside"):
        assert_windows_within_entities(store, _rebuilt(view.index, entity_codes=misowned))
    forged = SignalExamples(store, _rebuilt(view.index, entity_codes=misowned))
    with pytest.raises(SplitError, match="outside"):
        resolve_window_masks(forged, split, split.fold(0))


def test_an_index_group_that_contradicts_the_store_is_refused(store: SignalStore) -> None:
    view = _view(store)
    split = _kfold(store)
    groups = list(view.index.entity_groups)
    groups[0] = "someone-else"
    forged = SignalExamples(store, _rebuilt(view.index, entity_groups=groups))

    with pytest.raises(SplitError, match="group"):
        resolve_window_masks(forged, split, split.fold(0))


def test_an_index_entity_the_store_lacks_is_refused(store: SignalStore) -> None:
    view = _view(store)
    split = _kfold(store)
    names = list(view.index.entity_names)
    names[0] = "ghost"
    forged = SignalExamples(store, _rebuilt(view.index, entity_names=names))

    with pytest.raises(SplitError, match="ghost"):
        resolve_window_masks(forged, split, split.fold(0))


def test_an_entity_id_shaped_like_a_window_id_is_refused(tmp_path: Path) -> None:
    digest = WindowSpec(length=4, stride=2).digest
    names = [f"p{index}" for index in range(11)] + [f"p0:0:{digest}"]
    store = _store(tmp_path / "ambiguous", entity_ids=names)
    split = _kfold(store)

    with pytest.raises(SplitError, match="window id"):
        resolve_window_masks(_view(store), split, split.fold(0))


def test_a_colon_in_an_entity_id_alone_is_fine(tmp_path: Path) -> None:
    names = [f"site:{index}:raw" for index in range(12)]
    store = _store(tmp_path / "colons", entity_ids=names)
    split = _kfold(store)

    masks = resolve_window_masks(_view(store), split, split.fold(0))

    assert masks["train"].any()


def _without_group(view: SignalExamples, group: str) -> SignalExamples:
    return view.subset(view.groups != group)


def test_an_unexpected_absent_group_is_refused(store: SignalStore) -> None:
    split = _kfold(store)
    fold = split.fold(0)
    lost = sorted(fold.parts["train"])[0]

    with pytest.raises(SplitError, match=lost):
        resolve_window_masks(_without_group(_view(store), lost), split, fold)


def test_a_named_absent_group_is_accepted_and_another_is_not(store: SignalStore) -> None:
    split = _kfold(store)
    fold = split.fold(0)
    first, second = sorted(fold.parts["train"])[:2]
    view = _without_group(_view(store), first)

    masks = resolve_window_masks(view, split, fold, absent_groups=[first])

    assert first not in set(view.groups[masks["train"]].tolist())
    with pytest.raises(SplitError, match=second):
        resolve_window_masks(
            _without_group(view, second), split, fold, absent_groups=[first, "unrelated"]
        )


def test_an_empty_role_is_refused_even_when_its_groups_are_named(store: SignalStore) -> None:
    split = _kfold(store)
    fold = split.fold(0)
    view = _view(store)
    train_only = view.subset(np.isin(view.groups, fold.parts["train"]))

    with pytest.raises(SplitError, match="no windows"):
        resolve_window_masks(train_only, split, fold, absent_groups=fold.parts["test"])


def test_absent_groups_must_not_be_a_bare_string(store: SignalStore) -> None:
    split = _kfold(store)

    with pytest.raises(SplitError, match="absent_groups"):
        resolve_window_masks(_view(store), split, split.fold(0), absent_groups="p0")


def test_the_existing_gates_still_refuse_a_window_view(store: SignalStore) -> None:
    view = _view(store)
    split = _kfold(store)

    with pytest.raises(SplitError, match="unknown sample identity"):
        validate(view, split)
    with pytest.raises(SplitError, match="unknown sample identity"):
        resolve_masks(view, split, split.fold(0))


def test_evidence_reports_what_the_fold_actually_holds(store: SignalStore) -> None:
    split = _kfold(store)
    fold = split.fold(0)
    lost = sorted(fold.parts["train"])[0]
    view = _without_group(_session(_view(store), 0), lost)

    masks = resolve_window_masks(view, split, fold, absent_groups=[lost])
    evidence = window_evidence(view, split, fold, masks)

    assert evidence["split"] == "people"
    assert evidence["split_digest"] == split.digest
    assert evidence["fold"] == 0
    assert evidence["view_spec"] == view.index.digest
    assert evidence["view_derivation"] == view.derivation
    train = evidence["roles"]["train"]
    assert train["windows"] == int(masks["train"].sum())
    assert train["groups"] == len(fold.parts["train"]) - 1
    assert train["absent_groups"] == [lost]
    assert evidence["roles"]["test"]["absent_groups"] == []


def test_the_evidence_fingerprint_follows_the_windows(store: SignalStore) -> None:
    split = _kfold(store)
    fold = split.fold(0)
    view = _view(store)

    def fingerprint(examples: SignalExamples) -> str:
        masks = resolve_window_masks(examples, split, fold)
        return str(
            window_evidence(examples, split, fold, masks)["roles"]["train"]["window_ids_sha256"]
        )

    assert fingerprint(view) == fingerprint(_view(store))
    assert fingerprint(view) != fingerprint(_session(view, 0))


def test_resolving_leaves_the_manifest_digest_alone(store: SignalStore) -> None:
    split = _kfold(store)
    before = split.digest

    resolve_window_masks(_view(store), split, split.fold(0))

    assert split.digest == before


def test_a_window_at_the_top_of_the_integer_range_is_refused(store: SignalStore) -> None:
    view = _view(store)
    highest = np.iinfo(np.int64).max - (view.index.spec.length - 1)
    starts = view.index.starts.copy()
    starts[0] = highest

    with pytest.raises(ViewError, match="outside"):
        assert_windows_within_entities(store, _rebuilt(view.index, starts=starts))
