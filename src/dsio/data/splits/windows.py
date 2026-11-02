"""Replay a split over a store's entities on a window view of that store.

A split generated over :func:`~dsio.data.adapters.entity_examples` assigns recordings before
any window length exists, so it outlives every view taken of it. Its exact assignments name
entities and a view's examples are windows, which :func:`~dsio.data.splits.validation.validate`
and :func:`~dsio.data.splits.resolve.resolve_masks` rightly refuse: they compare exact sample
identities and they do not change here.

This module is the separate path, and it stays a small one. The split still passes the
unchanged gate, against the entities of the store the view is bound to. DSIO builds that
population itself rather than accepting one, because a caller-supplied table can carry
matching ids and a matching digest without being those entities. Only then does each window
inherit the role of the entity that owns it, which is safe because windows are proven to lie
inside their entities and entities are proven to partition.

A view may be filtered more tightly than the split's population; that is the point of it.
``view.derivation`` is therefore not compared with the split's: one is a cut of windows, the
other a cut of entities. A window of an entity the split never assigned is refused instead,
and a whole group lost to the filter must be named by the caller.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np

from dsio.contracts import sha256_of
from dsio.data.adapters import SignalExamples, entity_examples
from dsio.data.examples import Examples, ExamplesError, assert_consistent
from dsio.data.splits.models import SplitError, SplitFile, SplitFold
from dsio.data.splits.validation import validate
from dsio.data.store import SignalStore
from dsio.data.views import ViewError, assert_index_matches_store, assert_windows_within_entities


def resolve_window_masks(
    view: SignalExamples,
    split: SplitFile,
    fold: SplitFold,
    *,
    absent_groups: Collection[str] = (),
) -> dict[str, np.ndarray]:
    """Boolean mask per role over ``view``, for a split computed on its store's entities.

    Losing an entity to a filter is normal: a view of one activity has no recordings of
    another. Losing a whole group silently changes what a fold trains or tests on, so each
    expected loss is named in ``absent_groups`` and any other one is refused. A role with no
    windows at all is refused regardless, because it cannot be trained or scored.
    """
    if isinstance(absent_groups, str):
        raise SplitError("absent_groups must be a collection of group ids, not one string")
    if fold != split.fold(fold.index):
        raise SplitError(
            f"split {split.name!r} fold {fold.index} does not match the fold stored in the manifest"
        )
    if split.coverage != "total" or any(
        candidate.temporal is not None for candidate in split.folds
    ):
        raise SplitError(
            f"split {split.name!r} restricts examples by time, which belongs to the view it "
            "was computed on; only a total group split can be replayed across views"
        )
    try:
        assert_index_matches_store(view.store, view.index)
        assert_windows_within_entities(view.store, view.index)
        _assert_entities_are_not_window_ids(view.store)
        validate(_population(view.store, split, fold), split)
        assert_consistent(view)
    except (ExamplesError, ViewError) as error:
        raise SplitError(str(error)) from error

    window_ids = np.asarray(view.sample_ids, dtype=str)
    if np.unique(window_ids).size != window_ids.size:
        raise SplitError(f"view {view.name!r} repeats a window identity")

    entity_ids = np.asarray(view.index.entity_ids, dtype=str)
    masks = {role: np.isin(entity_ids, fold.assignments[role]) for role in split.required_roles}
    owners = np.sum(list(masks.values()), axis=0)
    unowned = np.flatnonzero(owners != 1)
    if unowned.size:
        raise SplitError(
            f"split {split.name!r} fold {fold.index} leaves {unowned.size} window(s) of "
            f"{view.name!r} without exactly one owner, first in entity "
            f"{entity_ids[unowned[0]]!r}; the view holds entities the split never assigned"
        )

    groups = np.asarray(view.groups, dtype=str)
    expected = {str(group) for group in absent_groups}
    for role, mask in masks.items():
        if not mask.any():
            raise SplitError(
                f"split {split.name!r} fold {fold.index} role {role!r} has no windows in this view"
            )
        lost = sorted(_absent_groups(fold, role, groups[mask]) - expected)
        if lost:
            raise SplitError(
                f"split {split.name!r} fold {fold.index} role {role!r} loses {len(lost)} "
                f"group(s) in this view: {', '.join(lost[:5])}; name them in absent_groups "
                "if that is expected"
            )
    return masks


def window_evidence(
    view: SignalExamples,
    split: SplitFile,
    fold: SplitFold,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Summarise masks from :func:`resolve_window_masks` for the run record.

    A split says who was assigned and a view says which windows exist, so neither records
    what a fold actually held. The fingerprint does, exactly, for membership: it does not
    cover labels or payloads. This re-checks nothing, so pass it the masks that were used.
    """
    window_ids = np.asarray(view.sample_ids, dtype=str)
    groups = np.asarray(view.groups, dtype=str)
    roles = {
        role: {
            "windows": int(mask.sum()),
            "groups": int(np.unique(groups[mask]).size),
            "absent_groups": sorted(_absent_groups(fold, role, groups[mask])),
            "window_ids_sha256": sha256_of(sorted(window_ids[mask].tolist())),
        }
        for role, mask in masks.items()
    }
    return {
        "split": split.name,
        "split_digest": split.digest,
        "fold": fold.index,
        "view_spec": view.index.digest,
        "view_derivation": view.derivation,
        "roles": roles,
    }


def _population(store: SignalStore, split: SplitFile, fold: SplitFold) -> Examples:
    """The entities ``split`` was computed on, rebuilt from the store itself."""
    entities = entity_examples(store)
    if split.examples_derivation == entities.derivation:
        return entities
    # One-step subsets reproduce their derivation exactly, so the unchanged gate that follows
    # accepts this only when the assignments really are that subset of this store.
    ids = np.asarray(entities.sample_ids, dtype=str)
    return entities.subset(np.isin(ids, sorted(fold.all_assignments)))


def _assert_entities_are_not_window_ids(store: SignalStore) -> None:
    """Refuse the one way a window-level split could pass as an entity-level one.

    A manifest does not say whether its identities name entities or windows, and saying so
    would change the digest of every committed manifest. An entity called
    ``<other entity>:<start>:<digest>`` is the only thing that makes the two indistinguishable.
    """
    names = {entity.entity_id for entity in store.entities}
    for name in sorted(names):
        pieces = name.rsplit(":", 2)
        if len(pieces) == 3 and pieces[0] in names and pieces[1].isdecimal():
            raise SplitError(
                f"entity {name!r} has the form of a window id of entity {pieces[0]!r}, so a "
                "split over this store cannot prove its assignments name entities"
            )


def _absent_groups(fold: SplitFold, role: str, present: np.ndarray) -> set[str]:
    return set(fold.parts[role]) - set(present.tolist())
