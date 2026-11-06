# Declared splits: replaying a published train/test division

Date: 2026-11-06. Status: design, implemented on `feat/declared-split`.

## Problem

Some corpora ship a fixed division that every published result uses: a `train/` and a `test/`
folder, a list of held-out ids. Reproducing such a result needs exactly that division. DSIO cannot
express it today:

- `generate()` only computes splits (scikit-learn group splitters, purged walk-forward), so it
  cannot produce the published one.
- `validate()` refuses any algorithm outside `STABLE_ALGORITHMS`, so a hand-written manifest with
  `algorithm: manual` never reaches `DsioDataModule`.

The first consumer is GaitID's reproduction of Zou et al. 2020 on whuGAIT Dataset #1, which ships
33,104 training and 3,740 test windows of 118 people (GaitID D10).

## Decision

Add one algorithm, `declared`, to the closed dispatcher. It reads each example's role from a named
per-example attribute instead of computing it, and records where the division comes from.

```python
manifest = generate(
    examples,
    "declared",
    name="whugait-d1-as-released",
    seed=0,
    parameters={
        "attribute": "released_role",
        "source": {"citation": "...", "url": "...", "sha256": "<archive digest>"},
    },
)
```

- **Roles.** Every attribute value must be one of the two `roles`, and each role must get at least
  one example. A third value is refused rather than dropped, so `declared` always has total
  coverage.
- **No seed.** Nothing is random, and `seed` must be 0. A non-zero seed would suggest that a rerun
  could differ.
- **Source.** `citation`, `url` and `sha256` are required non-empty strings. `sha256` is 64
  lowercase hex characters naming the file the division was read from. They are part of the
  normalized parameters, so they enter the manifest digest: the same assignments read from a
  different archive are a different split.
- **One fold**, index 0. There are no scikit-learn dependency entries, because none is used.
- **Validations** are the grouped set: identity membership, role disjointness, group
  disjointness, total coverage. A group that straddles the published roles is refused at
  generation. A corpus whose published split is within-group (the same person in train and test)
  must use a finer group key, such as one group per window, and state that choice. DSIO does not
  silently allow a group leak.

## Replay

`validate()` repeats the derivation. It renormalizes the parameters, refusing a manifest whose
stored parameters are not canonical or which has a seed. It requires exactly one fold. It then
checks that each role's assignments equal the examples whose attribute names that role. A manifest
edited after generation, or replayed over examples whose attribute changed, is refused with
"assignments disagree with the declared attribute". The existing identity, coverage, group and
count checks still run unchanged.

## Not changed

The `SplitFile` schema (`dsio.split/3`), `resolve_masks`, the window-view path and the split
evidence functions are unchanged. `manual` is still refused by `validate()`.

## Admission

`declared` is a stable dispatcher entry, not an experimental component. It has no model or
numerical behaviour, only a replay of an attribute. That follows the window-view precedent (PR
#100), where a change to the closed split surface went through a spec and an ADR 0006 amendment.
The first downstream use is GaitID P4.
