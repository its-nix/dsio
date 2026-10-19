---
baseline_commit: 9727fa37f0fa8dd9076e96e7a85312d2ba9328b9
---

# Story 9.3: Balance Training by Class and Group

Status: done

## Story

As a consumer with imbalanced classes or participants,
I want DSio to fit class-frequency and group-frequency weights on the training role and apply them through the weighting rule,
so that balancing needs no local code (cohort #4b, #4c).

## Acceptance Criteria

1. Given a store-bound split manifest with a declared training role, when class weights are fitted, then only training-role labels contribute; the result uses `N / (K * class_count)`, records the declared zero-based class domain, counts, weights, governed identities, resolved entity identities when applicable, role, partition, and source policy, and rejects missing classes or malformed labels with an actionable error.
2. Given window assignments whose labels live on owning entities, when class weights are fitted, then governed windows are resolved through the bound `SignalExamples` index and each entity contributes exactly once in stable first-seen order; foreign indexes, unknown or duplicate assignments, and inconsistent entity labels are rejected before fitting.
3. Given training-role sample/window groups, when group weights are fitted, then each training sample receives `N / (G * group_count)`, the weights are finite and positive, their training-role mean is one, each group has equal total contribution, and the evidence records counts, group weights, sample weights, governed identities, role, and partition.
4. Given fitted group sample weights are injected into `WindowItems`, when a training assignment is materialized, then every assigned identity receives its fitted weight; a disjoint held-out assignment receives neutral weight one; partial overlap, unknown fitted identities, non-finite/non-positive values, or configuration/runtime collisions fail explicitly. `WindowItems` performs no frequency fitting.
5. Given fitted class weights are applied by a native loss and class plus group weights are combined, then unweighted-sample objectives preserve the native loss reduction, sample-weighted objectives compute `mean(group_weight * native_per_sample_loss)`, and no micro-batch-local normalization or weighted sampler is introduced.
6. Given flat Child Mind and Child Mind Sequence are migrated, when training runs, then both `_class_weights` helpers and dataset-side participant-frequency calculation are deleted; both fitted mappings are logged through stable `fitted/<name>.json` artifacts and included in provenance, with fitted values injected as runtime arguments where components consume them.
7. Given the completed migration, when contract and governance gates run, then both consumers match committed golden metrics exactly, the CMI-sequence micro-batch invariance regression passes, the two experimental fitters have real-use evidence without automatic promotion, the class/group fitting candidate is removed, deletion metrics improve, no dependency or golden changes occur, and all repository gates pass.

## Tasks / Subtasks

- [x] Task 1: Add manifest-authoritative class-frequency fitting (AC 1-2).
  - [x] Add one generic public fitter in `dsio.experimental.data.fitting`, with an explicit positive class count and named attribute source.
  - [x] Reuse the established store/digest/training-role guards. Support store-sample attributes and window-to-unique-entity attributes through bound `SignalExamples` without accepting caller-owned training IDs.
  - [x] Preserve stable identity order; validate zero-based integral labels, complete declared class coverage, finite positive output, and canonical evidence.
- [x] Task 2: Add manifest-authoritative group-frequency fitting (AC 3).
  - [x] Fit over governed training assignments, not unique entities: sequence windows belonging to the same participant remain separate weighted samples.
  - [x] Return deterministic group counts, per-group weights, and per-training-sample weights. Validate exact identity coverage and mean-one normalization.
  - [x] Prove held-out group/label mutations cannot affect either fitter.
- [x] Task 3: Inject fitted sample weights without dataset-side fitting (AC 4-5).
  - [x] Replace `WindowItems`' `inverse_group_frequency` mode with a fitted-weight field and runtime mapping.
  - [x] Require complete coverage for a fitted/training assignment, emit neutral ones for a wholly disjoint role, and reject partial overlap. Preserve reorderings and worker determinism.
  - [x] Delete `Counter` and all frequency arithmetic from `WindowItems`; retain finite-positive scalar validation at the application boundary.
- [x] Task 4: Lock fitting and weighting semantics with focused tests (AC 1-5).
  - [x] Cover formulas, ordering, complete/missing classes, malformed labels, unequal groups, equal total group contribution, mean one, duplicate/unknown/empty assignments, foreign store/index, canonical serialization, and validation-before-read behavior.
  - [x] Cover flat sample labels and window-to-entity label deduplication separately; prove group fitting intentionally does not deduplicate windows.
  - [x] Cover fitted mapping application, neutral held-out weights, partial overlap rejection, shuffled assignments, spawn workers, and runtime/config collision.
  - [x] Retain native class-weight and combined sample-weight loss equivalence plus the existing micro-batch partition regression.
- [x] Task 5: Migrate flat Child Mind class fitting (AC 1, 5-7).
  - [x] Fit four-class weights from training-role `target` attributes, record the fitted artifact/mapping, inject only the weight vector into the objective, and include the evidence in provenance/return data.
  - [x] Delete the local `_class_weights` helper and unused NumPy import without changing datasets, model architecture, objective semantics, split topology, or downstream behavior.
- [x] Task 6: Migrate Child Mind Sequence class/group fitting (AC 2-7).
  - [x] Fit four-class weights over unique training entities using `label`; fit participant group weights over training windows without deduplication.
  - [x] Record both artifacts/mappings and inject the class vector into the objective plus sample-weight mapping into the configured `WindowItems` runtime argument, including calibration-created data modules.
  - [x] Delete `_training_entities`, `_class_weights`, and assignment-local participant weighting while preserving validation execution with neutral held-out weights.
- [x] Task 7: Close evidence, parity, and repository gates (AC 6-7).
  - [x] Add per-callable experimental evidence for the single CMI real-use group, remove the candidate, regenerate the catalog, and pass admission/static audits without consumer names in DSio source.
  - [x] Run both CMI contract/replay/export tiers, exact golden assertions, sequence micro-batch invariance, and fitted provenance/artifact assertions.
  - [x] Record deletion metrics against the completed Story 9.2 baseline; keep goldens, dependencies, and lockfiles unchanged.
  - [x] Run focused and full pytest (explicit live marker deselected), build, Ruff lint/format, mypy, distribution/consumer-flow contracts, and all import contracts.

### Review Findings

- [x] [Review][Patch] Inject fitted class weights as runtime component arguments and record a separate provenance config so configured/runtime collisions remain enforced [reference_projects/kaggle/child_mind/tasks/training.py:130]
- [x] [Review][Patch] Generalize group fitting to manifest-governed sample assignments as well as windows, while retaining window-level non-deduplication [src/dsio/experimental/data/fitting.py:438]
- [x] [Review][Patch] Harden fitter identity boundaries: validate examples name/digest, window owner/group consistency, duplicate governed identities, impossible class domains, and float32-emitted weights [src/dsio/experimental/data/fitting.py:73]
- [x] [Review][Patch] Require explicit fitted-weight runtime injection when a fitted `sample_weight` field is configured; downstream no-weight reads must opt into an empty neutral mapping [src/dsio/experimental/data/windows.py:86]
- [x] [Review][Patch] Add the missing malformed-input, foreign/duplicate/unknown assignment, held-out isolation, fitted mapping, and runtime/config collision regressions [tests/experimental/data/test_fitting.py:48]
- [x] [Review][Patch] Include fitted class/group evidence in consumer return data as required by the implementation tasks [reference_projects/kaggle/child_mind/tasks/training.py:211]

## Dev Notes

### Implementation seam

- Extend `src/dsio/experimental/data/fitting.py`, which already owns authoritative training-role selection and `record_fitted`. Factor private manifest/index resolution helpers if that reduces duplication; do not create consumer callbacks or accept an arbitrary list of training IDs.
- Recommended public names are `fit_class_weights` and `fit_group_weights`. Both remain experimental because the two CMI variants count as one unrelated real-use group.
- Class output schema should include `counts`, `weights`, `classes`, `attribute`, `source`, `role`, `partition`, `sample_ids`, and `resolved_sample_ids`. Group output should include `counts`, `group_weights`, `sample_weights`, `role`, `partition`, and `sample_ids`. Mapping keys and identity order must serialize deterministically.
- Sequence class fitting deduplicates owning entities. Sequence group fitting does not: every training window is a loss sample, and `weight(group) = N / (G * count(group))`.
- `WindowItems` should accept fitted sample weights as a runtime-only argument selected by its recorded field configuration. At construction/materialization, an assignment must have either complete fitted coverage or no overlap; no overlap emits neutral ones for held-out roles. Mixed coverage is a configuration error.
- Class weights remain native objective parameters. Flat CMI keeps native `cross_entropy(..., weight=...)`; sequence keeps `cross_entropy(..., weight=..., reduction="none")` followed by `mean(sample_weight * loss)`.

### Scope boundaries

- Story 9.4 owns fusion/encoder composition and classifier/objective deletion. Story 9.5 owns subgroup/ablation evaluation. Story 9.6 owns representative parity. Do not pull them forward.
- Do not add weighted samplers, streaming fitters, registries, new dependencies, task/project dispatch, multi-input models, or per-batch weight normalization.
- Do not change ingestion, feature meaning, split topology, architecture, output shape, participant aggregation, submission formatting, standardization behavior, committed goldens, or dependency files.
- Preserve the completed but uncommitted Story 9.1 and 9.2 work in the current worktree.

### Previous-story intelligence

- Story 9.1 established `WindowItems`, canonical provenance, group identity, digest verification, and worker determinism. Its assignment-local inverse-group mode was explicitly temporary until this story.
- Story 9.2 established store/manifest authority, window-to-entity stable deduplication, complete fitted evidence, canonical JSON, runtime collision protection, and the fit → record → inject lifecycle. Reuse those guards and patterns.
- Story 7.4 established native class-weight reduction and mean-one sample-weight semantics. Mathematical rewrites can change reduction order and exact goldens, so retain the current consumer objective code in this story.
- Completed Story 9.2 baseline: flat CMI 720 non-ingestion lines, sequence CMI 901, Kaggle total 5,570, banned local definitions 11; full gate 1,358 passed and one explicit live test deselected.

### Verification order

1. Focused fitter and `WindowItems` tests, then objective weighting tests.
2. Flat CMI contract/export/replay/golden tests; sequence contract/export/replay/micro-batch/golden tests.
3. Catalog/admission/metrics and unchanged golden/dependency diffs.
4. Distribution/consumer-flow contracts and full pytest with only the explicit live marker deselected.
5. Build, Ruff lint/format, mypy, and three import contracts.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Epic 9 / Story 9.3]
- [Source: `_bmad-output/planning-artifacts/prd.md` — FR6]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-4 and constraints]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — blocks #4b and #4c]
- [Source: `_bmad-output/specs/spec-component-warehouse/conventions.md` — weighting, construction, and evidence rules]
- [Source: `_bmad-output/specs/spec-component-warehouse/brownfield.md` and `roadmap.md` — parity and scope]
- [Source: `docs/component-admission.md`]
- [Source: `_bmad-output/implementation-artifacts/7-4-train-with-dsio-objectives.md`]
- [Source: `_bmad-output/implementation-artifacts/9-1-map-windows-into-training-items-with-weights-and-groups.md`]
- [Source: `_bmad-output/implementation-artifacts/9-2-fit-observed-only-statistics-for-missing-data.md`]
- [Source: `src/dsio/experimental/data/fitting.py`, `windows.py`, and `model/objectives.py`]
- [Source: both Child Mind `components.py` and `tasks/training.py` modules]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- Story created from the canonical Warehouse SPEC/companions, Epic 9 contract, current implementation, and completed Stories 7.4, 9.1, and 9.2.
- Implementation plan: add manifest-governed fitters and focused failures first; then replace dataset-side fitting with injected sample weights; finally migrate both consumers, update evidence, and run parity/full gates.

### Completion Notes List

- Added manifest-authoritative class and group fitters with stable evidence, strict identity/label validation, train-only fitting, sample/window support, and deterministic mean-one weighting.
- Replaced `WindowItems` frequency fitting with explicit runtime sample-weight injection, neutral held-out behavior, full/none coverage enforcement, float32-safe values, and collision protection.
- Migrated flat and sequence CMI training to fit, record, inject, and return class/group evidence while preserving native loss semantics, calibration, replay, export, and goldens.
- Adversarial review closed 6 patches; 10 reported items were dismissed after triage and no findings were deferred.
- Final metrics: flat CMI 711, sequence CMI 909, Kaggle total 5,569 non-ingestion lines, and 11 governed local definitions (Story 9.2 baseline: 720/901/5,570/11).
- Verification: 1,370 passed, 1 explicit live test deselected, 188 warnings; Ruff lint/format, mypy (124 files), three import contracts, wheel/sdist build, unchanged goldens, and unchanged dependency files all passed.

### File List

- `_bmad-output/implementation-artifacts/9-3-balance-training-by-class-and-group.md` (new)
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `_bmad-output/planning-artifacts/epics-component-warehouse.md`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/evidence.yaml`
- `reference_projects/kaggle/child_mind/tasks/training.py`
- `reference_projects/kaggle/child_mind/sequence/components.py`
- `reference_projects/kaggle/child_mind/sequence/tasks/evaluation.py`
- `reference_projects/kaggle/child_mind/sequence/tasks/training.py`
- `src/dsio/experimental/data/__init__.py`
- `src/dsio/experimental/data/fitting.py`
- `src/dsio/experimental/data/windows.py`
- `tests/experimental/data/test_fitting.py`
- `tests/experimental/data/test_window_items.py` (new)
- `tests/kaggle_portfolio/test_child_mind.py`
- `tests/kaggle_portfolio/test_child_mind_sequence.py`

## Change Log

- 2026-10-18: Created an implementation-ready story with explicit class-domain, participant-deduplication, train-only group application, evidence, parity, and governance contracts.
- 2026-10-18: Implemented class/group fitting and runtime weight injection, migrated both CMI consumers, closed all adversarial review patches, and passed the full repository gate suite.
