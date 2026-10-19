---
baseline_commit: 9727fa37f0fa8dd9076e96e7a85312d2ba9328b9
---

# Story 9.2: Fit observed-only statistics for missing data

Status: done

## Story

As a consumer with missing values,
I want DSio to fit and apply standardization over observed values only, both fitted and per-window,
so that missingness never corrupts normalization.

## Acceptance Criteria

1. Given governed features containing NaNs or paired with a declared observation mask, when fixed statistics are fitted, then DSio derives identities exclusively from the manifest's declared training role; excludes every unobserved value before arithmetic; computes finite float64 per-feature population mean and standard deviation; applies the consumer-declared small-scale rule; reports every all-missing feature by name or index; and held-out-value mutation cannot change the result.
2. Given assignments that name stored entities or indexed windows, when statistics are fitted from stored data columns or entity attributes, then source/store/index bindings are validated before reads; window assignments are mapped to their owning entities and deduplicated in stable first-seen order; missing, duplicate, foreign, or malformed identities fail explicitly; and the fitted result records role, partition, selected identities, source selection, observation mode, mean, and scale.
3. Given fixed standardization with a declared feature-level observation mask, when it is applied, then observed finite values are normalized, unobserved positions become exact zero, masked non-finite values do not enter arithmetic, observed non-finite values fail, mask/presence channels remain available to the enclosing single-`x` model, and existing unmasked `Standardize` behavior remains compatible.
4. Given per-instance signal standardization with either a declared shared validity channel or feature-specific NaNs, when it is applied, then each sample/feature is normalized over its observed timesteps only; missing positions become exact zero; zero-variance and all-missing windows have deterministic declared behavior; dtype, device, reduction mode, compilation, and replay behavior remain supported; and the existing shared validity channel is preserved unchanged.
5. Given a training run using fitted statistics, when evidence and provenance are recorded, then `record_fitted` writes a stable run-relative artifact; provenance references that artifact and the complete fitted mapping; fitted values are injected as runtime component arguments; configured/runtime collisions still raise; and export, MLflow load, evaluation, ablation, and submission reuse the fitted values without refitting.
6. Given `child_mind` and `child_mind/sequence`, when migrated as one vertical path, then `_statistics`, `_observed_statistics`, `_tabular_statistics`, `_participant_entities`, `_normalize_observed`, `_normalize_window`, and their normalization parameter helpers (`_vector`, `_scale`, `_binary`/`_binary_mask`) are deleted; the top-level model still receives only `x`; both contract tiers, replay identity, export behavior, and golden metrics remain unchanged.
7. Given the completed migration, when governance and repository gates run, then the real uses and standing approval are recorded, statistics are removed from the candidate ledger while Story 9.3 weight-fitting remains, catalog/admission checks pass, deletion metrics improve, no dependency or golden file changes are introduced, and all repository gates pass.

## Implementation Slices

1. **Generic fitting contract (AC 1-2).** Extend the existing `fit_standardization`; do not add a parallel CMI fitter. Add declarative value/observation selection for stored columns and indexed entity attributes, with stable entity deduplication and named features. Establish exact arithmetic in focused tests before consumer edits.
2. **Observed-only application (AC 3-4).** Extend `Standardize` for an explicit feature-level observation mask and `InstanceStandardize` for feature-specific finite-value observation. Preserve their current public modes and NumPy-parity path.
3. **Consumer vertical migrations (AC 5-6).** Fit, record, inject, apply, export, and evaluate through the generic blocks in both CMI flows. Delete the named local helpers only after exact focused golden/replay parity passes.
4. **Admission and closure (AC 7).** Update evidence/candidates/catalog, measure deletion, run adversarial review, and complete full gates.

## Tasks / Subtasks

- [x] Task 1: Extend the existing training-role fitter without duplicating policy (AC 1-2).
  - [x] Keep `fit_standardization(store, manifest, *, partition, ...)` as the sole public statistics fitter. Preserve its current default behavior and result keys for Bike Sharing compatibility.
  - [x] Add declarative selection of value columns plus either NaN observation or same-width explicit observation columns. Select masked values before arithmetic so masked NaN/Inf values are harmless; reject observed non-finite values.
  - [x] Support values stored in a named entity attribute when governed assignments are window IDs. Require the bound `SignalExamples`/index, validate its store binding, resolve each assigned window to its owning entity, and deduplicate entities in stable first-seen order before reading attributes.
  - [x] Keep the manifest's first required/training role authoritative. Reject empty, duplicate, unknown, foreign, or malformed assignments before fitting; never accept caller-supplied held-out identities as authority.
  - [x] Accept optional feature names and identify all all-missing features in one actionable error. Record source kind/name, selected value and observation columns, observation mode, resolved identities, role, and partition in the fitted mapping.
  - [x] Make the zero/small-scale fallback explicit and validated. Default to existing `scale == 0 -> 1`; allow the CMI migration to preserve its current `< 1e-6 -> 1` rule without embedding consumer names or policy in DSio.
- [x] Task 2: Lock fitting semantics with focused tests (AC 1-2).
  - [x] Cover default fitting compatibility, NaN exclusion, explicit feature masks, masked non-finite values, observed non-finite rejection, named all-missing errors, float64 population statistics, stable identity order, and configured small-scale fallback.
  - [x] Mutate held-out data and prove fitted mean/scale and training identities do not change.
  - [x] Cover window-to-entity resolution and participant deduplication, plus foreign index/store, unknown window/entity, duplicate assignment, missing attribute, malformed widths/columns/masks, and validation-before-read failures.
  - [x] Preserve `record_fitted`'s stable `fitted/<name>.json` path and canonical finite JSON payload.
- [x] Task 3: Extend fixed observed-only application (AC 3).
  - [x] Extend `Standardize` rather than creating a second fixed-standardization component. Add an explicit observed-only mode accepting a same-shape feature mask at the internal model-composition boundary; keep one-input/unmasked calls backward compatible.
  - [x] Validate mean/scale/axis and mask shape/binary values centrally. Require positive scale for newly fitted observed mode while preserving any documented compatibility needed by the existing mode.
  - [x] Apply selection before arithmetic, normalize only observed finite values, and emit exact zero for unobserved values. Do not widen the top-level Predictor/model input beyond `x`.
  - [x] Test broadcasting/axis, masked NaN/Inf, observed NaN/Inf, binary/shape errors, dtype/device, state-dict buffers, configuration round-trip, compilation, and unchanged ordinary `Standardize` behavior.
- [x] Task 4: Extend per-instance observed-only application (AC 4).
  - [x] Extend `InstanceStandardize` with a mutually exclusive feature-specific finite/NaN observation mode; retain the current shared `observed_channel` mode and `[B,C,T]` contract.
  - [x] Compute each feature from its own observed timesteps, population variance, and configured epsilon/scale floor; zero missing positions. Define all-missing feature behavior as deterministic zero for per-instance application (the fit-time all-missing error does not apply to an individual padded window).
  - [x] Preserve the NumPy reduction mode used for audited consumer parity and update it for feature-specific masks without changing existing shared-mask results.
  - [x] Test mixed feature masks, completely missing features/windows, absent modality, zero variance, masked and observed non-finite values, dtype/device, compile, state/config round-trip, deterministic replay, and existing Parkinson FoG behavior.
- [x] Task 5: Migrate flat Child Mind fitting and application (AC 5-6).
  - [x] Fit tabular and sensor-summary statistics from the manifest training role using their existing packed value/mask slices. Preserve float64 sum/square population results and `<1e-6 -> 1` semantics exactly.
  - [x] Record each run's fitted artifact and full mapping in provenance, and inject the mean/scale values through runtime component arguments so collision protection remains active.
  - [x] Compose the still-local classifier with DSio `Standardize` blocks for its two value/mask pairs. Preserve packed layout, masks, presence behavior, encoder inputs, buffer/state-dict compatibility required by the flow, and the single top-level `x` input.
  - [x] Delete `_statistics`, `_observed_statistics`, `_normalize_observed`, `_vector`, `_scale`, and `_binary_mask` only after flat flow/export/replay/golden tests pass.
- [x] Task 6: Migrate Child Mind Sequence fitting and application (AC 5-6).
  - [x] Fit packed tabular entity attributes from training window assignments through the bound index. Preserve unique-participant weighting of statistics; never count a participant once per window.
  - [x] Record and inject the fitted mapping as in the flat flow. Keep window mapping, class/group weighting, objective, participant aggregation, and model architecture unchanged; those belong to Stories 9.3-9.5.
  - [x] Compose the local classifier with DSio `Standardize` for tabular value/mask pairs and `InstanceStandardize` in feature-specific finite mode for raw sensor windows. Preserve time-major packing, channel order, modality presence, NumPy-parity arithmetic, and Conv1d inputs exactly.
  - [x] Delete `_tabular_statistics`, `_participant_entities`, `_normalize_window`, `_vector`, `_scale`, and `_binary` only after sequence flow/export/replay/micro-batch/golden tests pass.
- [x] Task 7: Prove lifecycle, provenance, and consumer parity (AC 5-6).
  - [x] Assert provenance contains artifact, role, partition, resolved identities, selection configuration, observation mode, mean, and scale; paths remain run-relative and replay-stable.
  - [x] Assert downstream export/evaluation/ablation/submission rebuild from recorded component config and checkpoint state without fitting or reading held-out values.
  - [x] Retain the runtime/config collision regression and Predictor export -> MLflow load -> predict identity for affected component paths.
  - [x] Run both CMI contract tiers, replay tests, sequence micro-batch invariance, and unchanged exact golden assertions before full tests.
- [x] Task 8: Close governance and quality gates (AC 7).
  - [x] Add the CMI real-use group to `fit_standardization`, `record_fitted`, `Standardize`, and `InstanceStandardize` evidence as applicable. Treat `child_mind` plus `child_mind/sequence` as one unrelated-use group and do not auto-promote any component.
  - [x] Split/update the candidate entry so only Story 9.3 class/group weight fitting remains. Regenerate `catalog.md` and pass admission/static audits; DSio source/docstrings must contain no consumer names.
  - [x] Run consumer metrics and record the deletion delta without rewriting the Story 6.6 baseline. Keep `tests/golden_metrics.json`, dependencies, and lockfiles unchanged.
  - [x] Run focused tests, catalog/admission/metrics, distribution and consumer-flow contracts, full default pytest (explicit live marker deselected), build, Ruff lint/format, mypy, and all import contracts.

### Review Findings

- [x] [Review][Patch] Reject non-finite fitted means/scales produced by overflow before artifact or model construction [src/dsio/experimental/data/fitting.py:238]
- [x] [Review][Patch] Record complete source, observation, identity, threshold, reduction, and feature-name policy for every non-default fit [src/dsio/experimental/data/fitting.py:256]
- [x] [Review][Patch] Honor the declared reduction for fully observed data instead of silently using NumPy standard deviation [src/dsio/experimental/data/fitting.py:246]
- [x] [Review][Patch] Reject zero-width/malformed sources and validate data-column configuration before governed payload reads [src/dsio/experimental/data/fitting.py:17]
- [x] [Review][Patch] Validate the actual float32 registered mean/scale buffers after conversion to prevent underflow or overflow [src/dsio/experimental/model/standardization.py:103]
- [x] [Review][Patch] Reject non-finite per-instance statistics or normalized outputs caused by finite-value overflow [src/dsio/experimental/model/standardization.py:285]
- [x] [Review][Patch] Add missing governance, malformed-input, lifecycle, broadcasting/device, compilation, and overflow regressions for the new fitting and observed-only modes [tests/experimental/data/test_fitting.py:139]

## Dev Notes

### Current implementation seam

- `src/dsio/experimental/data/fitting.py` already owns training-role selection, store/manifest validation, float64 population fitting, NaN-aware mode, all-NaN rejection, and `record_fitted`. Extend that path; do not introduce `fit_observed_statistics` or consumer callbacks.
- `fit_standardization` currently assumes manifest assignments are store sample IDs. Sequence assignments are window IDs, so entity-attribute fitting must resolve them through the governed `SignalExamples` index and deduplicate owners. This is required for parity, not an optimization.
- `Standardize` owns fixed mean/scale buffers and broadcasting. `InstanceStandardize` owns rank-3 per-instance normalization, a shared validity-channel mode, Torch and NumPy reductions, and exact-zero invalid outputs. Add modes to these blocks instead of adding overlapping public components.
- `resolve_component` already rejects configured/runtime collisions. Follow the Story 7.2 pattern: fit -> `record_fitted` -> add artifact and complete mapping to provenance -> inject only fitted constructor values at resolution.

### Numerical and missingness invariants

- Fitted statistics use population variance (`ddof=0`) and float64 accumulation. Missing selection happens before arithmetic; multiplying NaN by zero is not valid masking.
- Flat CMI currently derives variance from sums and squares and maps scales below `1e-6` to one. Sequence per-window normalization uses population variance and a `1e-6` floor. Preserve these results exactly; mathematically equivalent reduction order can still change logits and goldens.
- Fit-time all-missing features are data/configuration errors and must be named. Per-window all-missing features are expected for padding or absent modalities and must deterministically emit zeros.
- Flat CMI has separate tabular and sensor value/mask slices plus a presence bit. Sequence signal missingness is feature-specific (`isfinite(value) & sensor_present`), not one shared timestep channel.
- A batch `mask` is not a model input. All model-side validity remains inside the declared packed `x` channels or is derived from declared NaN padding.

### Scope boundaries and dependency traps

- Story 9.3 owns class weights, participant/group weights, and final weighting semantics. Do not migrate `_class_weights` or change the preserved Story 9.1 assignment-local weighting here.
- Story 9.4 owns the pooled temporal encoder, multimodal fusion, classifier deletion, objective/output migration, and model factory. The local classifiers remain and should only compose admitted standardizers in this story.
- Story 9.5 owns subgroup/ablation evaluation; Story 9.6 owns representative parity recording. Do not pull either forward.
- Do not change ingestion, feature meaning, split topology, model architecture, training flow, objectives, output shape, participant aggregation, or submission formatting.
- Do not add dependencies, registries, arbitrary callbacks, project-name branches, multi-field model inputs, weighted samplers, or streaming fit behavior.
- The worktree contains completed but uncommitted Story 9.1 changes in sequence files and governance artifacts. Patch the current contents and preserve those edits.

### Previous-story intelligence

- Story 9.1 established configured `WindowItems`, canonical dataset provenance, group identity, process-local digest verification, and current sequence file shapes. Its review fixed group-keyed weighting, entity ownership/bounds, metadata-length validation, and provenance gaps; do not regress them.
- Story 7.2 established manifest-authoritative training-only fitting, run-relative fitted evidence, full mapping in provenance, and runtime component injection. Reuse that lifecycle.
- The last complete gate reported 1,346 passed, one explicit live test deselected, 124 mypy source files, three import contracts, unchanged goldens, and 927 Child Mind Sequence non-ingestion lines.

### Project Structure Notes

- Core fitting changes belong in `src/dsio/experimental/data/fitting.py`; update `src/dsio/experimental/data/__init__.py` only if public exports actually change.
- Application changes belong in `src/dsio/experimental/model/standardization.py`; update `src/dsio/experimental/model/__init__.py` only if public names change. Prefer no new public names.
- Focused contracts belong in `tests/experimental/data/test_fitting.py` and `tests/experimental/model/test_signals.py` (or a narrowly named standardization test file if separation materially improves clarity).
- Consumer edits are expected in both CMI `components.py` and training task modules. Downstream/evaluation task modules should change only when required to assert recorded provenance; they must not refit.
- Governance edits belong in `docs/component-warehouse/evidence.yaml`, `candidates.yaml`, and generated `catalog.md`; update sprint status and this story artifact last.

### Verification order

1. Fitting and standardization focused unit tests, including existing Bike and Parkinson compatibility tests.
2. Flat CMI flow/export/replay/golden tests, then sequence flow/export/replay/micro-batch/golden tests.
3. Component catalog, admission coverage, consumer metrics, and unchanged golden diff.
4. Distribution/consumer-flow contracts and full default pytest with only the explicit live marker deselected.
5. Build, Ruff lint/format, mypy, and three import contracts.

### Technical notes

- NumPy `nanmean`/`nanstd` emit warnings and return NaN for all-NaN slices; detect counts explicitly so the fitter raises one named deterministic error instead of relying on warnings. NumPy standard deviation is population-based by default (`ddof=0`).
- PyTorch registered buffers remain the correct storage for fixed fitted tensors because buffers follow device moves and persistent buffers participate in `state_dict`.
- Installed constraints already cover the work: Python `>=3.12,<3.15`, NumPy `2.5.2`, PyTorch `2.13.0+cpu`, Lightning `2.6.5`, and MLflow `3.16.1`. No dependency update is required.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Epic 9 / Story 9.2 and adjacent scope]
- [Source: `_bmad-output/planning-artifacts/prd.md` — FR7 and component-warehouse outcomes]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-4, constraints, non-goals]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — blocks #4a, #9a, #9b]
- [Source: `_bmad-output/specs/spec-component-warehouse/conventions.md` — component configuration, evidence, admission, and single-`x` conventions]
- [Source: `_bmad-output/specs/spec-component-warehouse/brownfield.md` and `roadmap.md` — migration boundaries and ordering]
- [Source: `docs/component-admission.md` — real-use, approval, test, provenance, and promotion rules]
- [Source: `_bmad-output/implementation-artifacts/7-2-fit-standardization-on-training-role-samples.md`]
- [Source: `_bmad-output/implementation-artifacts/9-1-map-windows-into-training-items-with-weights-and-groups.md`]
- [Source: `src/dsio/experimental/data/fitting.py`]
- [Source: `src/dsio/experimental/model/standardization.py`]
- [Source: `reference_projects/kaggle/child_mind/components.py` and `tasks/training.py`]
- [Source: `reference_projects/kaggle/child_mind/sequence/components.py` and `tasks/training.py`]
- [Source: NumPy `nanmean`: https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html]
- [Source: NumPy `nanstd`: https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html]
- [Source: PyTorch `Module.register_buffer`: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- RED/GREEN core: expanded `fit_standardization`, `Standardize`, and `InstanceStandardize` contracts; 46 focused fitting/signal tests passed after edge hardening.
- Consumer vertical verification: CMI flat/sequence plus Bike and Parkinson compatibility, governance, and metrics passed 116 tests with unchanged committed goldens.
- Adversarial review: Blind Hunter, Edge Case Hunter, and Acceptance Auditor produced seven accepted patches; all were applied and regression-tested.
- Full gate: 1,358 passed, one explicit live test deselected, 188 warnings; build, Ruff lint/format, mypy over 124 source files, three import contracts, catalog/admission, consumer metrics, and golden diff all passed.

### Completion Notes List

- Extended the sole training-role fitter with stored-column and indexed entity-attribute sources, explicit observation masks, stable participant deduplication, named all-missing errors, selectable population-reduction parity, small-scale policy, complete evidence, and overflow/canonical-JSON guards.
- Extended `Standardize` with same-shape observed-only masks and `InstanceStandardize` with feature-specific non-finite omission while preserving their existing modes, state/config behavior, devices, compilation, and NumPy-parity reduction.
- Migrated both Child Mind flows to fit, record, inject, checkpoint, export, reload, and evaluate through DSio blocks; deleted all Story 9.2 local fit/normalization/validation helpers while retaining the single packed `x` contract and exact golden metrics.
- Recorded the CMI real-use evidence without promotion and reduced the candidate ledger to Story 9.3 class/group weight fitting.
- Deletion measure from the completed Story 9.1 baseline: flat Child Mind 761 -> 720 non-ingestion lines (-41), Child Mind Sequence 927 -> 901 (-26), Kaggle total 5,637 -> 5,570 (-67); banned local definitions remain 11 and belong to later stories.
- Review hardening rejects zero-width or malformed sources before payload reads, non-finite fitted/per-instance results, and float32 parameter underflow/overflow; every non-default fitting policy is reconstructible from provenance.

### File List

- `_bmad-output/implementation-artifacts/9-2-fit-observed-only-statistics-for-missing-data.md` (new)
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/evidence.yaml`
- `reference_projects/kaggle/child_mind/components.py`
- `reference_projects/kaggle/child_mind/tasks/training.py`
- `reference_projects/kaggle/child_mind/sequence/components.py`
- `reference_projects/kaggle/child_mind/sequence/tasks/training.py`
- `src/dsio/experimental/data/fitting.py`
- `src/dsio/experimental/model/standardization.py`
- `tests/experimental/data/test_fitting.py`
- `tests/experimental/model/test_signals.py`
- `tests/kaggle_portfolio/test_child_mind.py`
- `tests/kaggle_portfolio/test_child_mind_sequence.py`

## Change Log

- 2026-10-18: Created an implementation-ready story from the Epic 9 contract, current code seams, prior migration lessons, and current NumPy/PyTorch behavior.
- 2026-10-18: Implemented observed-only fitting/application, migrated both CMI consumers, recorded governance/deletion evidence, and preserved exact contract/golden parity.
- 2026-10-18: Applied all seven adversarial review patches, passed the full repository gate, and completed the story.
