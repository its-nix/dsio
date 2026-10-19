---
baseline_commit: 9727fa37f0fa8dd9076e96e7a85312d2ba9328b9
---

# Story 9.1: Map windows into training items with weights and groups

Status: done

## Story

As a consumer training on windows,
I want the DSio window dataset to use the same field mapping as stored items, including per-window `sample_weight` and `group`,
so that I never write a window Dataset.

## Acceptance Criteria

1. Given a `SignalExamples` window index and a field-mapping `ComponentConfig`, when assigned windows are read, then the dataset emits the governed `sample_id`, exactly the configured tensor fields, and the configured `group` identity; preserves assignment order; verifies the index/store binding and the owning sample payload digest; and fails by field/sample name on invalid or missing configured input.
2. Given window, entity-attribute, index-label/metric, group, layout, dtype, transform, shape, and composite-input mappings, when a field is produced, then it follows the same validation and transform semantics as `StoredItems`; `x` can concatenate declared parts without consumer-specific code; `sample_weight` is finite, positive, scalar, and normalized to split-wide mean one when assignment-local inverse-group-frequency weighting is selected.
3. Given Child Mind Sequence training, export, and streaming evaluation, when the consumer is migrated, then `CmiSequenceWindows` and `sequence_windows` are deleted; the dataset is instantiated from a canonical `ComponentConfig`; full configuration is recorded in provenance; participant identity comes from `group`; sensor presence is read from its declared `x` channel rather than a duplicate item field; and contract metrics, replay identity, micro-batch invariance, export behavior, and golden metrics remain unchanged.
4. Given the admitted window-mapping component, when governance checks run, then its catalog entry contains all required sections, its real use and approval are recorded in `evidence.yaml`, the Window Dataset candidate is removed, the legacy `WindowDataset` disposition is explicit, current consumer metrics record the deletion delta, and all repository gates pass without changing golden values.

## Implementation Slices

1. **Generic window-to-item path (AC 1-2).** Implement and unit-test one public experimental factory from `SignalExamples`/`WindowIndex` to identity-bearing map-style items. Keep the existing store/index and `IdentityDataset` guards intact. Leave every gate green before touching the consumer.
2. **Child Mind Sequence vertical migration (AC 3).** Configure the new factory, migrate training/export/evaluation together, delete only the local dataset class/factory, and prove exact contract/replay/golden parity. Story 9.1 remains in progress until this slice passes.
3. **Admission and measurement closure (AC 4).** Update evidence/candidates/catalog, record the local-definition and non-ingestion-line delta, run adversarial review, and complete the full gates.

## Tasks / Subtasks

- [x] Task 1: Define the smallest generic window mapping contract (AC 1-2).
  - [x] Add one public module-level factory, tentatively `WindowItems`, under `dsio.experimental.data`; keep its constructed `Dataset` private.
  - [x] Reuse/refactor `StoredItems` field validation and conversion instead of copying dtype, transform, shape, finite-value, and error logic.
  - [x] Resolve assigned governed IDs to index positions once; reject unknown/duplicate or mismatched identities explicitly; preserve caller order.
  - [x] Support generic declared sources required by the real use: window payload, entity attribute, index label/metric, group identity, and composite `x` parts. Do not add callbacks, registries, or consumer-name branches.
  - [x] Preserve `sample_id` as mandatory metadata. Emit no undeclared tensor field. `group` is the reserved identity field, not a tensor and not `participant_id`.
  - [x] Preserve the current participant-balancing formula for this story: for the assigned role, `weight(group) = N / (G * count(group))`; validate finite positive scalar values and mean one. Keep class/group fitting evidence and training-role-only fitters in Story 9.3.
- [x] Task 2: Make window reads integrity-preserving and efficient (AC 1-2).
  - [x] Call `assert_index_matches_store` through `SignalExamples`/`WindowIndex`; do not accept a generic `Examples` implementation.
  - [x] Verify the complete owning entity payload before serving any of its windows. Cache only verification state per dataset process/worker so repeated windows do not re-hash the entity; never treat the index digest alone as payload verification.
  - [x] Slice with entity-relative coordinates after verification, refuse boundary crossing, copy/contiguize before `torch.from_numpy`, and keep items on CPU.
  - [x] Preserve map-style `__len__`/`__getitem__`, spawn picklability, deterministic worker behavior, and bounded per-item/window memory. Do not cache complete entity arrays.
- [x] Task 3: Add focused contract and regression tests (AC 1-2).
  - [x] Cover exact field set/order, governed identity, reordered assignments, group output, labels/metrics/attributes, composite input ordering, columns/layout/dtype/transforms/shape, and assignment-local weights.
  - [x] Cover foreign index/store, unknown identity, missing attribute/metric/label, invalid source/spec, shape/column failure, non-finite/invalid weights, boundary errors, and errors naming field plus sample.
  - [x] Mutate an owning payload and prove the first window read fails its digest check; prove a second window of a verified entity avoids redundant verification without weakening the corruption test.
  - [x] Compare `num_workers=0` and spawn workers, including shuffled order, exact tensors, groups, weights, and relative store paths.
- [x] Task 4: Migrate Child Mind Sequence as one coherent vertical path (AC 3).
  - [x] Declare the complete dataset `ComponentConfig` in consumer configuration. Compose `x` in the existing byte/float order: packed tabular attribute, flattened time-major sensor window, then sensor-presence scalar; preserve float32, target scalar int64, and sample weight float32.
  - [x] Replace both training and evaluation `dataset_factory=sequence_windows`; use the resolved configured factory in `DsioDataModule` and record the identical full config under provenance components.
  - [x] Update export example creation to use the configured dataset. Update streaming evaluation to aggregate by `batch["group"]` and derive sensor presence from the declared final `x` channel.
  - [x] Delete `CmiSequenceWindows` and `sequence_windows`; retain model/objective/output/streaming-evaluation candidates for Stories 9.2-9.5.
  - [x] Preserve `CmiSequenceObjective`'s split-wide-mean-one behavior and its micro-batch invariance test. Do not pull Story 9.2/9.3 standardization or weight fitters into this story.
- [x] Task 5: Record evidence, deletion, and parity (AC 3-4).
  - [x] Add the real Child Mind Sequence use and standing Epic 7-9 approval to `docs/component-warehouse/evidence.yaml`; remove the Window Dataset entry from `candidates.yaml`; regenerate `catalog.md`.
  - [x] Decide and document the legacy `WindowDataset` disposition: preserve it only if remaining tests/fixtures are real compatibility evidence; otherwise migrate those uses and delete it under the pre-1.0 legacy rule. Do not expose two overlapping supported window APIs.
  - [x] Run `tools/consumer_metrics.py`; require Child Mind Sequence's dataset class/factory findings to disappear and record the non-ingestion-line delta without rewriting the Story 6.6 baseline.
  - [x] Keep `tests/golden_metrics.json` unchanged and run the Child Mind Sequence contract/replay flow before the full gate.

### Review Findings

- [x] [Review][Patch] Compute inverse-frequency weights by emitted group rather than entity code [src/dsio/experimental/data/windows.py:141]
- [x] [Review][Patch] Validate indexed entity ownership and entity-relative window bounds before serving the first verified read [src/dsio/experimental/data/windows.py:153]
- [x] [Review][Patch] Scope payload-verification state to the current worker process [src/dsio/experimental/data/windows.py:140]
- [x] [Review][Patch] Validate index label and metric array lengths before item access [src/dsio/experimental/data/windows.py:157]
- [x] [Review][Patch] Record the canonical dataset configuration in export and evaluation provenance [reference_projects/kaggle/child_mind/sequence/tasks/evaluation.py:63]
- [x] [Review][Patch] Add the missing acceptance-criteria regression coverage for shared groups, malformed ownership/bounds, mapping failures, relative paths, and worker verification [tests/experimental/data/test_window_items.py:18]
- [x] [Review][Patch] Keep synthetic index-label and metric values from shadowing legal entity attributes [src/dsio/experimental/data/windows.py:156]

## Dev Notes

### Current implementation seam

- `WindowDataset` already supplies raw windows, `sample_id`, `row`, optional labels, dtype conversion, index/store binding, map-style loading, and worker determinism, but it is legacy and has no real-use evidence. It does not map composite fields, emit `group`/weights, accept `ComponentConfig`, or verify the owning payload digest.
- `StoredItems` is the authoritative validation/transform behavior to reuse. Its private helpers currently live in `src/dsio/experimental/data/items.py`; extract a private shared module only if that prevents duplication without widening the public API.
- `SignalStore.read()` enforces bounds and entity boundaries but does not verify content. `read_sample()` verifies a complete entity digest. The new path must combine verified entity ownership with window slicing rather than claiming `assert_index_matches_store()` is a payload check.
- `DsioDataModule` still accepts a resolved callable factory. Follow the established Epic 7 pattern: resolve the dataset `ComponentConfig` in consumer wiring and record the original config verbatim in provenance. Changing the DataModule API is out of scope.
- Child Mind Sequence currently computes participant weights inside its local dataset and exposes duplicate `participant_id` and `sensor_present` fields. Its `x` already contains both the participant's packed tabular values and the final sensor-presence channel.

### Scope boundaries and dependency traps

- This story maps windows and preserves current assignment-local participant weighting. Story 9.3 owns training-role fitters, evidence logging, and final class/group-weight injection. Do not pre-implement those fitters here.
- Do not move observed-only normalization (9.2), the classifier/objective (9.3-9.4), output validation (9.4), or bounded-memory participant aggregation (roadmap) into this story.
- `group` is optional batch metadata and never a second model input. The single-`x` Predictor contract remains unchanged.
- Avoid arbitrary field callbacks, bare lambdas, runtime registries, project-specific label meaning, sampler behavior, or a warehouse-owned flow/task.
- Do not change goldens to make migration pass. Any mismatch is an implementation/reproducibility defect until proved otherwise.

### Previous-story intelligence

- Story 8.7 closed Epic 8 only after clean-commit, zero-tolerance representative parity and adversarial review. It records the latest full gate: build; 1,327 default tests with one live test deselected; Ruff lint/format; mypy over 124 source files; three import contracts; catalog/admission; unchanged goldens.
- Its deletion measurement leaves exactly 10 registered Epic 9 Child Mind candidates plus 3 explicit self-supervised fixtures. A fresh 2026-10-17 measurement reports 13 banned definitions and 5,673 Kaggle non-ingestion lines.
- The unmerged `refactor/clean-sequence-semantics` branch is one commit ahead of `main` and changes Parkinson signal semantics only. It does not overlap the Story 9.1 window/CMI files; do not silently fold it into this story.

### Technical notes

- Keep a map-style dataset: current PyTorch documentation defines it through `__getitem__` and `__len__`, and `DataLoader` supports deterministic single- and multi-process loading over that contract.
- NumPy memory maps are array-like and permit segment access, but slicing alone does not establish DSio payload integrity; DSio's own digest contract remains authoritative.
- No dependency or lockfile update is required.

### Project Structure Notes

- Primary source: `src/dsio/experimental/data/windows.py` or a narrowly named replacement beside `items.py`; export only the chosen public factory from `src/dsio/experimental/data/__init__.py`.
- Shared private field-mapping helpers may move under `src/dsio/experimental/data/`; preserve `StoredItems`' public import and behavior.
- Focused tests belong under `tests/experimental/data/`; retain `tests/dataset/test_dataset.py` only for any intentionally preserved legacy surface.
- Consumer changes are limited to `reference_projects/kaggle/child_mind/sequence/components.py` and its training/evaluation task modules plus directly affected tests/docs.

### Verification

Run in this order:

1. `uv run pytest -q tests/experimental/data/test_items.py tests/dataset/test_dataset.py <new-window-items-tests>`
2. `uv run pytest -q tests/kaggle_portfolio/test_child_mind_sequence.py tests/test_component_catalog.py tests/experimental/test_admission_coverage.py tests/test_consumer_metrics.py`
3. `uv run python tools/catalog.py` and `uv run python tools/consumer_metrics.py`
4. Distribution/consumer-flow contracts and the full default pytest suite; do not silently include the explicit live marker.
5. `uv build`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and `uv run lint-imports`.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Epic 9 / Story 9.1 and delivery rules]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-3, CAP-4, constraints, non-goals]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — block #2]
- [Source: `_bmad-output/specs/spec-component-warehouse/conventions.md` — batch fields, weighting rule, component configuration/admission]
- [Source: `_bmad-output/implementation-artifacts/8-7-record-sequence-migration-parity.md`]
- [Source: `src/dsio/experimental/data/items.py`]
- [Source: `src/dsio/experimental/data/windows.py`]
- [Source: `src/dsio/data/views.py` — `WindowIndex` and `assert_index_matches_store`]
- [Source: `src/dsio/data/store/reader.py` — `read` versus `read_sample`]
- [Source: `reference_projects/kaggle/child_mind/sequence/components.py`]
- [Source: `reference_projects/kaggle/child_mind/sequence/tasks/training.py`]
- [Source: `reference_projects/kaggle/child_mind/sequence/tasks/evaluation.py`]
- [Source: `tests/kaggle_portfolio/test_child_mind_sequence.py`]
- [Source: PyTorch `torch.utils.data` documentation: https://docs.pytorch.org/docs/stable/data.html]
- [Source: NumPy `memmap` documentation: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- Implementation plan: extend the experimental window module with one configured factory,
  reuse `StoredItems`' validated scalar/array conversion, verify each owning entity once per
  dataset process, then migrate Child Mind Sequence and close catalog/evidence gates.
- RED: the new window-item contract suite initially failed at import because `WindowItems`
  did not exist. GREEN: 50 focused data tests passed after the generic implementation.
- Admission initially rejected direct imports of private `StoredItems` helpers; importing the
  public module and sharing its private implementation internally preserved one validation
  path without adding catalog surface. Catalog/admission then passed.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Story creation was rebased from the stale `feat/6-6-baselines` checkout to `main`; completed Stories 7.1-8.7 were not recreated.
- Added configured `WindowItems` with composite field mapping, group identity, assignment-local
  mean-one weights, index/store validation, per-entity payload verification, and worker-safe
  map-style loading.
- Migrated Child Mind Sequence training, export, and streaming evaluation; removed its local
  dataset class/factory; provenance records the complete dataset config; golden/replay metrics
  remain unchanged.
- Preserved legacy `WindowDataset` explicitly as unsupported pre-1.0 legacy evidence for its
  existing compatibility tests; `WindowItems` is the sole admitted real-use window component.
- Deletion measure: Child Mind Sequence fell by 36 non-ingestion lines (963 to 927); repository
  banned definitions fell from 13 to 11. The immutable Story 6.6 baseline was not rewritten.
- Verification: 1,346 passed, 1 explicit live test deselected; build, Ruff lint/format, mypy
  (124 source files), three import contracts, catalog/admission, focused CMI replay/golden, and
  consumer-metrics gates passed. `tests/golden_metrics.json` is unchanged.
- Adversarial review resolved seven findings: group-keyed weighting, indexed ownership/bounds,
  process-local verification, parallel metadata lengths, stage provenance, regression coverage,
  and collision-free index metadata mapping.

### File List

- `_bmad-output/implementation-artifacts/9-1-map-windows-into-training-items-with-weights-and-groups.md` (new)
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/evidence.yaml`
- `reference_projects/kaggle/child_mind/sequence/components.py`
- `reference_projects/kaggle/child_mind/sequence/tasks/evaluation.py`
- `reference_projects/kaggle/child_mind/sequence/tasks/training.py`
- `src/dsio/experimental/data/__init__.py`
- `src/dsio/experimental/data/windows.py`
- `tests/experimental/data/test_window_items.py` (new)
- `tests/kaggle_portfolio/test_child_mind_sequence.py`

## Change Log

- 2026-10-18: Implemented the configured window-item component, migrated Child Mind Sequence,
  recorded admission/deletion evidence, and moved the story to review after all gates passed.
- 2026-10-18: Resolved all seven adversarial review findings, reran the full gate, and completed
  the story.
