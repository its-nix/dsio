# Window views of entity-level splits

Date: 2026-11-02. Status: design, revised after review rounds 1 and 2.

## Problem

ADR 0006 and `SignalExamples.digest` promise that a split outlives its views: generate over
`entity_examples(store)` before any window length exists, then use it with any view of the same
store. Since governed splits gained exact sample assignments, that is not true.

`fold.assignments` holds entity ids, a window view has window ids (`entity:start:spec_digest`), so
`validate(view, split)` raises "names unknown sample identity". `DsioDataModule` calls that gate and
reads `fold.assignments` as the ids to load, and `IdentityDataset` requires one item per id, so a
dataset factory cannot expand recordings into windows either.

The result: a split generated before a view exists cannot train through `DsioDataModule`. A
consumer holding one committed set of group-level folds over a store, and training on several
filtered window views of it, must replay that one assignment in all of them and keep its recorded
evidence, rather than generate per-view splits.

## Decision

Keep the existing gate exactly as it is. Add one narrow, separate path for `SignalExamples`, and let
`DsioDataModule` opt into it explicitly.

`validate`, `resolve_masks`, `_validate_split_binding`, the manifest schema and the split evidence
functions do not change. `validate(view, split)` and `resolve_masks(view, split, fold)` keep their
current behaviour for an entity-level split.

The caller never supplies the source population, and there is no generic protocol for it. The
population is `entity_examples(view.store)`, built by DSIO from the store the view is bound to.
Review showed that a caller-supplied table, or a structural adapter returning one, can carry literal
ids and a matching digest without being that store's entities. `SignalExamples` is the only view
today; a protocol can come when a second trusted adapter exists.

### `dsio/data/views.py`

`assert_windows_within_entities(store, index)` sits beside `assert_index_matches_store`. Vectorised
over intervals, it requires every entity named by the index to exist in the store with the same
group, and every window to lie inside its entity's rows.

### New module, `dsio/data/splits/windows.py`

```python
def resolve_window_masks(
    view: SignalExamples,
    split: SplitFile,
    fold: SplitFold,
    *,
    absent_groups: Collection[str] = (),
) -> dict[str, np.ndarray]:
    """Boolean mask per role over ``view``, for a split computed on its store's entities."""
```

It does, in order:

1. Refuses a fold that differs from the manifest's stored fold, as `resolve_masks` does.
2. Refuses a split with `coverage != "total"` or any fold with `temporal` bounds, whatever the
   algorithm is called. Those restrict windows by time and belong to the view itself.
3. Runs `assert_index_matches_store` and `assert_windows_within_entities`.
4. Refuses a store where an entity id has the form of a window id of another entity in that store
   (`<entity>:<integer>:<digest>`). The manifest cannot say whether its ids name entities or windows,
   and adding that would change the digest of every committed manifest. Such ids are the only way a
   window split can pass as an entity split, so they are refused rather than guessed about.
5. Builds the population: `source = entity_examples(view.store)`. If `split.examples_derivation`
   differs from `source.derivation`, the split is nested, so
   `source = source.subset(np.isin(np.asarray(source.sample_ids, dtype=str), sorted(fold.all_assignments)))`.
   This is one id per entity, not one per window.
6. Runs the unchanged gate: `validate(source, split)`. It checks store name, digest, derivation, total
   coverage in every fold, and role and group disjointness. A one-step subset reproduces the
   derivation exactly, so nested splits pass, and only when the assignments really are that subset
   of this store. A population reached through several subset steps is refused by the derivation
   check, as it is today. A window-level split fails on identity and must use the existing path.
7. Runs `assert_consistent(view)` and refuses duplicate window ids.
8. Builds one mask per role with `np.isin(entity_ids, fold.assignments[role])`, where `entity_ids` is
   the view's per-window entity id as strings.
9. Refuses unless every window has exactly one owner. A window with none belongs to an entity this
   split never assigned: this stops a nested split being applied to groups outside it.
10. Refuses a role with no windows, always, because an empty held-out role cannot be scored.
11. Refuses a role where a group in `fold.parts[role]` has no windows under its mask, unless that
    group is named in `absent_groups`. Presence is read from the masked windows, not from the
    index's entity table, which a subset keeps in full. Losing entities is normal (a view of one activity
    has no recordings of another). Losing a whole group silently changes who a fold trains or tests on, so
    the caller must name each expected absence and an unexpected one still fails.

`view.derivation` is deliberately not compared with `split.examples_derivation`. The first is a cut
of windows, the second a cut of entities. The hazard that field guards against is a split applied to
a population it was not computed for: step 6 covers the source and step 9 the view. A view filtered
more tightly than the split's population is the intended use.

```python
def window_evidence(
    view: SignalExamples, split: SplitFile, fold: SplitFold, masks: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Summarise masks from :func:`resolve_window_masks` for the run record. It re-checks nothing."""
```

It returns JSON-ready values: split name and digest, fold index, `view.derivation`, and for each role
the window count, group count, `absent_groups`, and `sha256_of` the role's sorted window ids. That
fingerprint identifies realised membership exactly. It does not cover labels or payloads.

### `DsioDataModule`

Two new keyword arguments:

```python
split_level: Literal["examples", "source"] = "examples",
absent_groups: Collection[str] = (),
```

`split_level="examples"` keeps today's behaviour exactly. Nothing is inferred from id strings: a
caller states that the split assigns the view's source.

With `split_level="source"`:

- `__init__` raises `LoadingError` unless `examples` is a `SignalExamples`.
- `setup` requires `examples.digest == store.identity`, calls `resolve_window_masks` instead of
  `validate`, and takes each role's sample ids as the sorted ids of the windows in its mask. The
  factory still receives the view and those ids, so `WindowItems` works unchanged.
- `setup` computes `window_evidence` from the same masks that gave those ids. If earlier loaders were
  built from a different result, it raises `LoadingError` rather than mix them.
- The `window_evidence` property is published only after a successful `setup`. It raises
  `LoadingError` before that and in `examples` mode.

`absent_groups` with `split_level="examples"` raises `LoadingError`.

The whole view is passed for every phase. A predict-only module maps `predict` to a role of that same
view rather than passing a view holding only held-out windows, which would leave another role empty.

### Evidence

No change to the tracking package. `record_split_evidence(run, entity_examples(store), split, ...)`
and `load_split_evidence(uri, entity_examples(store), consumer_run_id=...)` already work, so a
training Run reuses the split recorded when the folds were prepared, and the manifest stays one small
file. For a nested split, callers pass the same subset they generated it on, as today.

`DsioDataModule.window_evidence` gives the project what to put in its provenance config, and the
project must do so: two filters can otherwise share one split's evidence. Enforcing that in the
tracking layer is a separate change.

### Docs

- Correct `SignalExamples.digest`, whose docstring claims direct resolution.
- Amend ADR 0006 with a dated note.
- Add **View** to `CONTEXT.md` if the term is not defined.

## Alternatives rejected

- **Caller-supplied `source`, or a generic `ViewExamples` protocol.** Review showed both can be
  defeated with literal ids, because a runtime protocol is not a trust boundary.
- **Identity-domain metadata on the manifest.** It would change every committed digest.
- **Project to a new window-level manifest.** Additive, and it reuses every check, but the manifest
  lists every window once per fold: tens of megabytes per Run at a few hundred thousand windows. It also breaks reuse of
  recorded split evidence, since each view gets a new digest. ADR 0006 avoids naming windows.
- **Teach `validate` and `resolve_masks` a view mode.** It puts a mode switch and a derivation
  exemption inside the gate every consumer passes.
- **Generate one split per view.** Loses replay of the committed assignment and its evidence.
- **A dataset factory that expands recordings.** `IdentityDataset` forbids it, correctly.

## Known limits, not changed here

- `setup` after mutating `fold_index` or `examples` can mix cached loaders. This exists today.
- `SignalExamples.attribute` broadcasts in store order and would misalign on a compact entity table.
  Built indices and subsets keep the full table, so this path does not reach it.
- `derive()` hashes a positional mask only, so `validate` cannot tell two same-sized masks over
  different populations apart. This exists today.
- `window_evidence` is not enforced at the tracking layer. A project that omits it loses the record
  of which filtered population a Run used.
- Resolution costs O(windows) in Python strings, like `validate` today. It suits hundreds of thousands of
  windows, not the 42M-window case in ADR 0006.

## Non-goals

- `folds_from_splits` over views. No consumer needs it yet.
- Temporal splits over views.
- Any change to `generate`, the manifest schema or digests. Existing committed manifests must load
  and hash identically.

## Tests

- Whole view and filtered view through `resolve_window_masks`: masks partition the view, and
  `assert_no_row_overlap` passes on the subsets.
- Nested split resolves on a view of its subset, with more windows than entities and several folds;
  a view with windows outside it is refused.
- Each refusal has its own test: changed fold, temporal algorithm, grouped manifest with temporal
  bounds, partial coverage, window-level split, wrong store, misowned window, entity id shaped like
  a window id, group mismatch, unowned window, empty role, unexpected absent group.
- A named absent group is accepted, another absent group is still refused, and an empty role is
  still refused.
- `resolve_masks(view, split, fold)` and `validate(view, split)` still raise.
- `window_evidence`: counts, absent groups, and a fingerprint that changes with the filter.
- `DsioDataModule` with `split_level="source"`: a real `WindowItems` factory loads each role's
  windows, batches carry window ids, no group crosses roles, predict maps to a role, and the same
  split works on two different filtered views. `split_level="examples"` is unchanged.
- Changing the view between two `setup` calls raises `LoadingError`; repeating `setup` does not.
- `absent_groups` in `examples` mode, and `split_level="source"` on non-signal examples, raise
  `LoadingError`.
- A committed split fixture keeps its digest.
