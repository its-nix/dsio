---
baseline_commit: 9727fa37f0fa8dd9076e96e7a85312d2ba9328b9
---

# Story 9.5: Evaluate Subgroups and Modality Ablation

Status: done

## Story

As a consumer evaluating a multimodal model,
I want `dsio.eval` to report subgroup metrics and modality-ablation deltas over arrays I supply,
so that evaluation is configuration instead of consumer-owned prediction, metric, artifact, and MLflow wiring (cohort #29).

## Acceptance Criteria

1. Given the existing stable `dsio.eval.evaluate` function, when optional subgroup and ablation declarations are omitted, then its source-compatible behavior, plain `dict[str, float]` result, validation order, native MLflow lineage, dataset-attributed metrics, and `evaluation/predictions.npz` contract remain unchanged. This story adds no parallel evaluator, result dataclass, registry, callback/plugin surface, project dispatch, or extra model input.
2. Given consumer-supplied subgroup masks and subgroup metric aliases, when evaluation validates them, then each subgroup is a named, non-empty, one-dimensional boolean NumPy array with one value per evaluation row; masks may overlap and need not partition the rows; metric aliases and generated keys are unique, MLflow-safe, within MLflow limits, and cannot collide with full-evaluation or ablation keys. Domain meaning and mask derivation remain consumer-owned.
3. Given a named ablation declaration containing an existing input field, a non-empty half-open last-axis slice, a selected subgroup, a finite scalar fill value, and metric aliases, when evaluation validates it, then the input is a numeric NumPy array with the evaluation row count and at least two axes; slice indices are non-boolean integers satisfying `0 <= start < stop <= input.shape[-1]`; the subgroup and metrics exist; and every detectable failure occurs before inference or MLflow evaluation writes.
4. Given a valid ablation, when evaluation runs, then it subsets every row-aligned input and the targets to the declared subgroup, copies the selected input without mutating caller arrays, fills only the declared last-axis slice, calls the same immutable Predictor once for that ablation, and verifies returned sample identity/order and configured prediction/score fields through the same shape, dtype, finiteness, and metric boundaries as the full prediction. Other input fields, dimensions, dtypes, and values remain unchanged.
5. Given full, subgroup, and ablated predictions, when evidence commits, then one flat mapping contains every base, subgroup, ablated, and delta metric; every value is finite and logged in one native MLflow metric batch with dataset name/digest attribution; normalized subgroup/ablation configuration is recorded within MLflow limits; and full outputs, subgroup masks/identities, ablated outputs, and exact ablation sample identities are persisted together under `evaluation/predictions.npz`. All validation, prediction, metric, and source-revalidation failures occur before persistence begins and leave no metrics, native inputs, or evaluation artifacts. Persistence uses MLflow's independent writes with metrics submitted last as the completion marker; a persistence failure raises `EvaluationError` but may leave incomplete non-metric evidence because MLflow provides no transaction or reliable artifact rollback across those writes.
6. Given metric alias `qwk`, subgroup names `sensor_present`/`sensor_missing`, and ablation name `sensor`, then generated keys are `qwk.sensor_present`, `qwk.sensor_missing`, `qwk.sensor_ablated`, and `qwk.sensor_ablation_delta`. Delta is explicitly **full subgroup metric minus ablated metric**, preserving current code, contract goldens, and representative baselines. This corrects the earlier epic wording; no computation or tolerance change is authorized.
7. Given flat `child_mind`, when migrated, then its task calls `dsio.eval.evaluate` once with caller-derived presence masks and packed last-axis sensor slice `[SENSOR_VALUE_SLICE.start, SENSOR_PRESENT_INDEX + 1]`; the extra full `predict`, local QWK calls, manual four-metric logging, and `ablate_sensor` helper/export are deleted. Exact metric keys/values, present and ablation sample IDs, provenance, replay identities, submission bytes, and both tabular/fused contract goldens remain unchanged.
8. Given `child_mind/sequence`, when Story 9.5 closes, then its bounded-memory window-to-participant aggregation, subgroup scoring, and modality ablation remain consumer-local and behaviorally unchanged, and the candidates register names that deferred surface truthfully. Whole-array `evaluate` does not absorb streaming, aggregation, or `ablate_sequence_sensor`.
9. Given the completed change, when governance and repository gates run, then no new public evaluator or experimental callable requires admission evidence; the catalog/candidate register is current; `dsio.eval` still imports no `data`, `model`, `train`, `tracking`, or experimental package; flat non-ingestion lines and Kaggle total improve from Story 9.4 baselines 684 and 5,539, sequence remains 906 except justified formatting, banned definitions remain 3, committed goldens/dependencies/lockfile are unchanged, and all repository gates pass.

## Tasks / Subtasks

- [x] Task 1: Freeze current evidence and polarity (AC 1, 5-9).
  - [x] Add an unequal regression proving `full - ablated`, exact keys, plain-dict return, dataset-attributed metrics, and persistence before production changes.
  - [x] Use two slices: generic `evaluate` extension with focused tests; flat CMI migration plus governance closure. Keep the parent in progress until both pass.
  - [x] Preserve existing full evaluation and CMI goldens; amend before observing results if computation must change.
- [x] Task 2: Extend existing evaluation without a parallel API (AC 1-6).
  - [x] Add optional plain mapping/array arguments for subgroup masks, subgroup metric aliases, and named ablations to `src/dsio/eval/execution.py`; keep current callers compatible.
  - [x] Reuse metric dispatch and validation helpers. Validate names, masks, aliases, collisions, MLflow limits, input fields, row alignment, numeric dtype, slice bounds, subgroup references, fills, prediction fields, and score fields at the narrowest boundary.
  - [x] Compute full prediction once, reuse it for subgroups, and predict each ablation once. Subset row-aligned inputs consistently, copy before filling, preserve `sample_id`, and never mutate caller data.
  - [x] Generate `<alias>.<subgroup>`, `<alias>.<ablation>_ablated`, and `<alias>.<ablation>_ablation_delta`; delta is selected subgroup full minus ablated.
  - [x] Commit one coherent evidence set: configuration params, linked model/dataset inputs, full/subgroup/ablation arrays, and all metrics in the existing final batch. Preserve pre-write and source-revalidation guards.
- [x] Task 3: Cover boundaries, failure atomicity, and compatibility (AC 1-6).
  - [x] Test overlapping subgroups, selected-row ablation, exact call counts, no mutation, last-axis filling, dtype/shape/identity preservation, NPZ keys, and deterministic replay.
  - [x] Reject non-boolean/wrong-rank/wrong-length/empty masks; unsafe/duplicate/colliding names; unknown subgroup/metric/input; non-array/non-numeric/wrong-row/rank-deficient inputs; boolean/negative/empty/out-of-range slices; and non-finite/unrepresentable fills.
  - [x] Test malformed/identity-changing ablated outputs, metric errors/non-finite results, and source changes; assert metrics, native inputs, and artifacts remain absent on pre-commit failure.
  - [x] Exercise existing `mask`, `target_names`, `prediction_field`, and `score_field`, or explicitly reject ambiguous combinations with a documented limitation.
- [x] Task 4: Migrate flat Child Mind vertically (AC 6-7).
  - [x] Pass `sensor_present`/`sensor_missing`, alias `qwk -> quadratic_weighted_kappa`, and one present-row zero-fill sensor ablation to `evaluate`.
  - [x] Remove redundant prediction, local metrics, manual metric logging, Torch used only for ablation, and `ablate_sensor` from components and `__all__`.
  - [x] Keep sample-ID results and provenance truthful; inspect the native artifact for aligned ablated outputs.
  - [x] Run both modes through goldens, replay, export/load/predict, provenance, identity, and byte-identical submission assertions.
- [x] Task 5: Preserve sequence roadmap boundary and close governance (AC 8-9).
  - [x] Do not migrate or inline sequence streaming aggregation/ablation. Clarify its candidate entry to include participant aggregation, subgroup metrics, and modality ablation; retain `ablate_sequence_sensor`.
  - [x] Regenerate/check catalog and run catalog/admission/import-contract tests. Do not invent evidence for an additive spine capability.
  - [x] Measure against flat 684 / sequence 906 / Kaggle 5,539 / banned definitions 3 and record results.
  - [x] Run focused/full pytest, Ruff lint/format, mypy, build, distribution/consumer-flow contracts, three import contracts, and unchanged `tests/golden_metrics.json`, `pyproject.toml`, `uv.lock` diffs.

### Review Findings

- [x] [Review][Patch] Align the persistence-atomicity contract with MLflow's enforceable boundary — AC5 now guarantees a clean run for every pre-persistence failure, identifies metrics as the final completion write, and explicitly permits incomplete non-metric evidence after a remote persistence failure because MLflow offers no cross-write transaction or reliable artifact rollback.
- [x] [Review][Patch] Require an ablation alias to resolve to the same registered metric as its subgroup baseline [src/dsio/eval/execution.py:335]
- [x] [Review][Patch] Reject narrowing or otherwise non-round-trippable ablation fill values instead of silently wrapping them [src/dsio/eval/execution.py:490]
- [x] [Review][Patch] Reject byte-string `sample_id` arrays at validation rather than accepting them and failing during JSON serialization [src/dsio/eval/execution.py:507]

## Dev Notes

### Required API contract

- Extend `dsio.eval.execution:evaluate`; FR13 forbids a new DSio evaluate function. `dsio.eval` is a stable spine surface exempt from warehouse admission, so do not add an `experimental.eval` wrapper merely to satisfy cohort terminology.
- Use plain mappings/NumPy arrays. A suitable serializable shape is subgroup masks keyed by name; subgroup metrics keyed by short alias to registered metric name; and ablations keyed by name with `input_field`, `[start, stop]`, `subgroup`, `fill`, and alias-to-metric mapping. Names may improve; semantics and evidence keys bind.
- Subgroups operate on evaluation rows before existing element masks/named-target reshaping. Reject ambiguous axis combinations rather than silently broadcasting.
- Ablation evaluates only its subgroup, matching current CMI behavior. Subset each row-aligned input; never infer presence, slice, or modality from `x`.
- Persist deterministic collision-checked keys such as `subgroup__<name>` and `ablation__<name>__prediction__<field>`; never object arrays/pickle.
- Retain Story 4.4 transaction order: validate/compute, revalidate clean attempt/model/dataset, persist artifact and native inputs/params/tags, then metrics last.

### Polarity correction

- Earlier epic prose conflicted with implementation and goldens. The binding convention is `qwk.sensor_ablation_delta = qwk.sensor_present - qwk.sensor_ablated`. Fused contract example: `-0.038461538461538325 - 0.6938775510204082 = -0.7323390894819465`.

### Scope boundaries

- Preserve sequence streaming: load one PyFunc, stream windows, sum probabilities per participant, average before argmax, then participant metrics/artifact. Whole-array evaluation cannot reproduce this without the roadmap API.
- Do not change models, normalization, fitted statistics, weighting, training/export, splits, flow topology, submissions, or metric implementations.
- Story 9.6 owns representative-tier reruns/URIs; Story 10.5 owns broader downstream reduction.
- No dependencies, Prefect tasks/flows, result model, metric registry, arbitrary transform callbacks, multi-field model inputs, streaming, or window-to-entity aggregation.

### Existing seams and files

- `src/dsio/eval/execution.py`: reuse `_metric_names`, `_field_name`, `_metric_arrays`, `_validate_metric_arrays`, `_compute_metrics`, `_output`, `_require_clean_attempt`, immutable evidence checks, NPZ persistence, dataset-scoped `log_batch`.
- `src/dsio/eval/metrics.py`: reuse `compute` and `quadratic_weighted_kappa`; no metric addition.
- `reference_projects/kaggle/child_mind/tasks/downstream.py`: duplicate prediction/local metrics/manual logging are deletion targets.
- `reference_projects/kaggle/child_mind/components.py`: delete flat `ablate_sensor` only.
- Preserve `reference_projects/kaggle/child_mind/sequence/tasks/evaluation.py`, `sequence/components.py`, and `ablate_sequence_sensor`.
- Retain and clarify `docs/component-warehouse/candidates.yaml`; no evidence entry is expected for a spine function.

### Previous-story intelligence

- Story 9.4 moved flat assembly to `collate_arrays`, standardized `prediction`/`probability`, and retained ablation helpers for 9.5/roadmap. Its review requires explicit shape, dtype, identity, logged-metric, and catalog assertions.
- Story 9.4 closed with 1,390 passed, one live test deselected, 188 warnings; flat 684, sequence 906, Kaggle 5,539; banned definitions 3.
- Story 9.2/9.3 deferrals remain unrelated in `deferred-work.md`.

### Library and testing requirements

- Keep Python `>=3.12,<3.15`, MLflow `>=3,<4`, PyTorch `>=2.7,<3`, Lightning `>=2.5,<3`, TorchMetrics `>=1.7,<2`; no dependency change.
- Unit coverage: `tests/eval/test_execution.py`; flat integration: `tests/kaggle_portfolio/test_child_mind.py`; sequence regression unchanged.
- Preserve byte-identical goldens (SHA-256 `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530`), `pyproject.toml`, and `uv.lock`. Run catalog, consumer metrics, CI gates, and adversarial review.

### References

- [Source: `_bmad-output/planning-artifacts/prd.md` — authority]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-6, constraints]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — #29 and streaming candidate]
- [Source: `_bmad-output/specs/spec-component-warehouse/roadmap.md` — streaming evaluation]
- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — FR10, FR13, Story 9.5]
- [Source: `docs/component-admission.md`]
- [Source: `_bmad-output/implementation-artifacts/9-4-compose-multimodal-fusion-with-modality-presence-gating.md`]
- [Source: `src/dsio/eval/execution.py` and both Child Mind evaluation modules]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- Canonical requirements, complete epics, current evaluation/consumer code, Story 9.4, governance, and git history analyzed.
- Requirements, code/API, and testing/governance analyses ran in parallel and were reconciled here.
- Implementation plan: freeze full-minus-ablated evidence in unit tests; extend the existing evaluate transaction; migrate flat CMI; retain and clarify the sequence streaming candidate; run review and complete gates.

### Completion Notes List

- Extended the existing `evaluate` transaction with validated row subgroups, aliased metrics, selected-row last-axis ablations, full-minus-ablated deltas, identity checks, native metric attribution, and combined NPZ evidence.
- Migrated flat Child Mind to one generic evaluation call; deleted its local ablation helper, redundant prediction/QWK work, and manual MLflow metric logging while preserving both mode goldens and replay/submission behavior.
- Retained sequence bounded-memory participant aggregation and clarified its complete roadmap candidate surface.
- Consumer measures: flat 684 → 654, sequence remains 906, Kaggle 5,539 → 5,509, banned definitions remain 3.
- Verification: 1,402 passed, one explicit live test deselected; 152 focused tests passed; Ruff, formatting, mypy, catalog, build, eight distribution/consumer-flow contracts, and three import contracts passed. Goldens, `pyproject.toml`, and `uv.lock` hashes are unchanged.
- Review clarified the enforceable MLflow transaction boundary: all pre-persistence failures leave a clean run; remote persistence failures may leave incomplete non-metric evidence, with metrics submitted last as the completion marker.
- Review fixes require metric-consistent aliases, reject out-of-range integer fills before inference, and reject byte-string identities at the validation boundary. Thirteen targeted regression cases and 60 evaluator/flat-consumer tests pass; Ruff, format checking, and focused mypy pass.

### File List

- `_bmad-output/implementation-artifacts/9-5-evaluate-subgroups-and-modality-ablation.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `_bmad-output/planning-artifacts/epics-component-warehouse.md`
- `src/dsio/eval/execution.py`
- `reference_projects/kaggle/child_mind/components.py`
- `reference_projects/kaggle/child_mind/tasks/downstream.py`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `tests/eval/test_execution.py`
- `tests/kaggle_portfolio/test_child_mind.py`

## Change Log

- 2026-10-18: Created implementation-ready story; corrected delta polarity to preserve full-minus-ablated behavior.
- 2026-10-18: Implemented generic subgroup/ablation evaluation and migrated flat Child Mind; all development gates pass.
- 2026-10-19: Adversarial review aligned AC5 with MLflow's non-transactional persistence boundary.
- 2026-10-19: Applied all adversarial-review patches and closed the story.
