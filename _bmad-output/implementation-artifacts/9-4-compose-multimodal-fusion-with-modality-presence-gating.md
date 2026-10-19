---
baseline_commit: 9727fa37f0fa8dd9076e96e7a85312d2ba9328b9
---

# Story 9.4: Compose Multimodal Fusion with Modality-Presence Gating

Status: done

## Story

As a consumer with optional modalities,
I want a fusion composition that encodes declared slices of `x` with configured encoders, gates by declared presence channels, and fuses into a head,
so that missing-modality models are configuration only (cohort #10, #15).

## Acceptance Criteria

1. Given one `x` with exact per-sample `input_shape=[F]` or `[1,F]` and an ordered configuration of named modality slices, encoders, optional presence channels, static enabled flags, and a configured head, when the fusion model is resolved, then it is a native importable `nn.Module`; accepts `[B,*input_shape]` and, only when explicitly enabled for `input_shape=[F]`, the compatible `[B,1,F]` consumer form; flattens no other layout; rejects negative, empty, out-of-range, or overlapping half-open modality slices; forbids presence indices inside a modality slice; permits presence indices to be shared; requires every flattened feature to belong to a slice or declared presence index; validates unique names, encoder outputs, and finite binary presence; builds every encoder in declared order followed by the head; and records canonical ComponentConfig parameters.
2. Given an optional modality is absent for one sample or statically disabled and its raw payload remains valid under the consumer's declared ingestion/mapping contract, when arbitrary finite or non-finite payload values change, then the prediction is exactly unchanged. Input is selected to neutral values before the encoder and the encoded contribution is selected to zero after it with `torch.where`, so absent payloads cannot affect batch-dependent encoders and encoder bias/NaN cannot leak. Present-modality values can affect the prediction, and any declared presence scalar is appended to the fused representation exactly as configured. Domain placeholder/mask validity remains enforced at ingestion or mapping boundaries before model execution; fusion does not reinterpret invalid domain payloads.
3. Given a pooled temporal Conv1d encoder configured behind `TimeMajorToChannelFirst`, when it receives a valid signal, then it performs the declared ordered Conv1d/activation stages followed by temporal mean pooling and returns `[batch, features]`; rank/channel/time/configuration failures name the violated extent. Configuring two `Conv1d(kernel_size=9, stride=4, padding=4)` stages with GELU reproduces the sequence consumer's encoder exactly.
4. Given fixed observed-only tabular/sensor statistics or per-instance missing-value statistics, when branch encoders are composed from single-input stages, then existing `Standardize`/`InstanceStandardize` behavior is reused through explicit packed-value/mask and emit-observation configuration. The packed fixed stage accepts `[B,2F]` ordered as `[values, observed]` and emits `[normalized_values, original_observed]`; per-instance `emit_observed` accepts channel-first `[B,C,T]` and emits `[B,2C,T]` ordered as `[normalized_values, derived_finite_observed]`. Both reject malformed extents/axes. Missing values are selected out before arithmetic, old public modes remain compatible, and missingness math is not embedded in fusion.
5. Given the same seed and inputs, when legacy flat and sequence reference architectures are compared with their configured replacements, then parameter construction order, shapes, initial values, logits, native class-weight reduction, sample-weighted `mean(weight * per_sample_loss)`, and logged `accuracy` are exact. The flat `tabular` mode still constructs the sensor encoder and the same `2 * hidden + 1` head before statically gating sensor/presence to zero. No tolerance or computation change is authorized by this story.
6. Given both consumers are migrated, when training, calibration, export, evaluation, and replay run, then `CmiFusionClassifier`, `CmiSequenceClassifier`, `CmiObjective`, `CmiSequenceObjective`, both local prediction modules and validators, and flat `_arrays` are deleted. Models use the fusion/temporal configurations, objectives use `SupervisedObjective`, outputs use `MulticlassOutput(classes=4, scores=True, score_field="probability")` and its validator, and flat arrays use `collate_arrays`; the two ablation helpers and sequence streaming aggregation remain for Story 9.5.
7. Given fitted statistics and class weights target nested components, when roots are constructed, then generic named nested-runtime mappings route values through each destination's own `resolve_component` call: fusion routes parameters by modality name into its encoder, sequential compositions route parameters by stage index, and `SupervisedObjective` routes runtime loss parameters into the native loss. Unknown destinations and collisions at the root or nested destination raise. Runtime-only mappings are absent from static configs, while provenance records separate fully resolved model/objective configs. Sequence calibration factories construct fresh equivalent roots, the Predictor/export API still accepts only `x`, and output fields remain `prediction` and `probability` across the MLflow round trip.
8. Given the completed migration, when governance and repository gates run, then the pooled encoder and fusion composition are experimental with one shared CMI use group and explicit standing approval; all affected existing components gain truthful evidence; the four completed CMI candidate groups are removed; generated catalog/doctests and admission audits pass; banned definitions fall from 11 to 3; non-ingestion lines improve from flat 711 / sequence 909 / Kaggle 5,569; committed goldens, dependencies, and lockfiles are unchanged; and all repository gates pass.

## Tasks / Subtasks

- [x] Task 1: Freeze behavior and declare implementation slices (AC 1-8).
  - [x] Add seeded legacy-reference tests before production changes for flat fused/tabular and sequence architectures, including parameter order/value and logit equality.
  - [x] Record three vertical slices: generic fusion plus flat migration; pooled temporal encoder plus sequence migration; governance/candidate closure. Keep the parent story in progress until all pass.
  - [x] Treat exact contract/golden parity as binding. If initialization or computation cannot remain exact, stop and amend this story with the precise change and tolerance before observing migrated results.
- [x] Task 2: Add the generic multimodal fusion composition (AC 1-2).
  - [x] Add a focused experimental model module or extend `compositions.py`; use generic names, module-level importability, nested ComponentConfigs, ordered module construction, and one `x` input.
  - [x] Declare last-axis half-open slices as JSON arrays, optional scalar presence indices, per-modality static enabled flags, and whether presence enters the fused head input. Validate all configuration and runtime boundaries with modality-named errors.
  - [x] Enforce exact `[B,*input_shape]`, non-negative disjoint slices, presence-outside-slice, complete flattened-feature coverage, and the sole `[1,F]` singleton-flattening form; shared presence indices are allowed.
  - [x] Neutralize absent/disabled input before encoder execution and zero its encoded output afterward using selection, not multiplication. Validate encoder Tensor/rank/batch contracts and head compatibility.
  - [x] Cover mixed, all-present, all-absent-optional, statically disabled, mutation, gradient, dtype/device, resolution, determinism, and malformed configuration cases.
- [x] Task 3: Add the pooled temporal encoder and composition seams (AC 3-4).
  - [x] Add a configurable pooled Conv1d block in `dsio.experimental.model.convolution`; do not repurpose the incompatible legacy dilated/BatchNorm `Conv1dEncoder`.
  - [x] Reuse `TimeMajorToChannelFirst` explicitly. Prove seeded equality with Conv1d → GELU → Conv1d → GELU → temporal mean for the sequence configuration.
  - [x] Extend the existing standardization blocks only as needed for exact `[values, observed] → [normalized, observed]` packed fixed behavior and `[B,C,T] → [normalized C, derived observed C]` per-instance behavior. Preserve all existing modes and tests; do not put missingness or layout policy inside fusion.
  - [x] Add generic nested runtime-parameter routing to the sequential composition so fitted values reach a named/indexed nested stage through `resolve_component`, with unknown-target and collision failures.
  - [x] Re-export every public callable, provide all catalog docstring sections and executable examples, and add focused signal/standardization tests.
- [x] Task 4: Preserve objective accuracy while deleting local objectives (AC 5-7).
  - [x] Configure flat `SupervisedObjective` with native `CrossEntropyLoss(weight=...)`; configure sequence with the same native loss plus `sample_weighted=True`.
  - [x] Supply a generic stateless class-index accuracy metric compatible with the objective's default mean and injected `reduction="none"` modes, or an equivalently admissible generic mechanism. Preserve the `accuracy` name and exact flat/weighted-sequence values; do not use a stateful TorchMetric inside `SupervisedObjective`.
  - [x] Extend `SupervisedObjective` with runtime-only native-loss parameters routed through `_build`/`resolve_component`; reject configured/runtime collisions and unknown values. Retain separately recorded resolved configs and add collision plus loss/metric equivalence regressions.
- [x] Task 5: Migrate flat Child Mind vertically (AC 2, 5-7).
  - [x] Express tabular and sensor packed branches, observed-only standardization, Linear/ReLU encoders, presence gating/appending, and Linear/ReLU/Linear head as configuration.
  - [x] Preserve `[B,F]` and `[B,1,F]` inputs, binary presence validation, absent-modality independence, masked non-finite behavior, seeded construction order, and the `tabular` mode's statically disabled but still constructed sensor branch.
  - [x] Replace the objective and output/validator with warehouse components. Add an inputs-only dataset configuration, replace `_arrays` with `collate_arrays`, and preserve identity order, float32 `[N,1,PACKED_FEATURES]`, export example, evaluation, inference, and submission bytes.
  - [x] Delete the flat classifier, objective, prediction module, validator, and `_arrays`; retain `ablate_sensor` for Story 9.5.
- [x] Task 6: Migrate Child Mind Sequence vertically (AC 2-7).
  - [x] Configure the packed tabular branch, flattened time-major sensor slice, explicit unflatten/layout adapter, per-instance observed-only standardization with observation channels, exact strided temporal encoder, presence gating/appending, and GELU head.
  - [x] Preserve ingestion/mapping guarantees that absent sequence payloads are zero placeholders, finite/binary presence, population statistics, temporal mean location, participant weights, calibration factory freshness, micro-batch invariance, and streaming evaluation. Model fusion handles valid absence but does not own domain placeholder validation.
  - [x] Replace the objective and output/validator; delete the sequence classifier, objective, prediction module, and validator. Retain `ablate_sequence_sensor` and window-to-participant streaming aggregation for Story 9.5/roadmap.
- [x] Task 7: Close evidence, candidates, parity, and gates (AC 6-8).
  - [x] Add evidence/approval and admission tests for each new public callable; extend evidence for `SupervisedObjective`, `MulticlassOutput`, `MulticlassValidator`, `collate_arrays`, `TimeMajorToChannelFirst`, and standardizers actually used. Count both CMI paths as one unrelated use; do not promote automatically.
  - [x] Remove the completed local-model, local-objective, prediction-normalizer/validator, and flat evaluation-array candidates; regenerate the catalog and run all doctests/static audits.
  - [x] Run both CMI focused, contract, export/load/predict, replay, exact-golden, provenance, calibration, and micro-batch tiers. Verify the eight CMI banned definitions disappear and record deletion metrics against 711/909/5,569/11.
  - [x] Run focused and full pytest with only the explicit live marker deselected, Ruff lint/format, mypy, build, distribution/consumer-flow contracts, three import contracts, and unchanged `tests/golden_metrics.json`, `pyproject.toml`, and `uv.lock` diffs.

### Review Findings

- [x] [Review][Patch] Validate class-index metric dtype, non-empty batches, and target range [src/dsio/experimental/model/objectives.py:273]
- [x] [Review][Patch] Reject empty channel/time extents in instance standardization [src/dsio/experimental/model/standardization.py:310]
- [x] [Review][Patch] Preflight every pooled convolution stage so short time extents raise a named component error [src/dsio/experimental/model/convolution.py:87]
- [x] [Review][Patch] Make MulticlassOutput reject the empty batch its validator rejects [src/dsio/experimental/inference/outputs/multiclass.py:62]
- [x] [Review][Patch] Enforce exact rank-two packed-standardization input [src/dsio/experimental/model/standardization.py:218]
- [x] [Review][Patch] Preserve Standardize's float32 result instead of promoting it through a float64 observation mask [src/dsio/experimental/model/standardization.py:226]
- [x] [Review][Patch] Complete fusion isolation, singleton/shared-presence, gradient/device, and nested-routing regressions [tests/experimental/model/test_fusion.py:45]
- [x] [Review][Patch] Restore float64 input adaptation for both CMI fusion configurations [reference_projects/kaggle/child_mind/components.py:68]
- [x] [Review][Patch] Reject flat layouts other than `[B,F]` and `[B,1,F]` before fusion [reference_projects/kaggle/child_mind/components.py:99]
- [x] [Review][Patch] Restore positive hidden-width validation in the flat configuration factory [reference_projects/kaggle/child_mind/components.py:68]
- [x] [Review][Patch] Regenerate the catalog after the reviewed packed-input contract correction [docs/component-warehouse/catalog.md:1170]
- [x] [Review][Patch] Complete exact parameter/objective/logged-accuracy parity regressions [tests/kaggle_portfolio/test_child_mind.py:301]
- [x] [Review][Defer] Revisit neutral fallback policy for entirely unobserved fitted features [reference_projects/kaggle/child_mind/tasks/training.py:85] — deferred, pre-existing Story 9.2 policy
- [x] [Review][Defer] Consider materializing sequence dataset runtime weights in component provenance [reference_projects/kaggle/child_mind/sequence/tasks/training.py:208] — deferred, pre-existing Story 9.3 provenance design

## Dev Notes

### Required component contract

- Recommended names are `MultimodalFusion`, `PooledConv1d`, and a generic stateless `ClassIndexAccuracy`; final names may change only if their contracts remain distinct and catalog-friendly.
- A fusion modality needs: a unique identifier name; one non-empty non-negative half-open flattened slice; an encoder ComponentConfig; optional scalar presence index; static `enabled` (default true); and presence-to-head inclusion. Slices are mutually disjoint, presence cannot sit inside a slice, shared presence indices are allowed, and slices plus unique presence indices cover the full flattened input. Accept only the exact configured `[B,F]` or `[B,1,F]` shape.
- Construct modalities in declaration order and the head last. This is scientifically material because both CMI goldens depend on seeded PyTorch initialization order.
- Gate raw modality input before the encoder and gate the embedding afterward. Use `torch.where`; `NaN * 0` is still NaN. A disabled modality is still constructed so parameter allocation and head shape remain identical.
- The fusion block owns slicing, presence validation/gating, concatenation, and head dispatch only. It does not own normalization, class/group weights, domain placeholder rules, ablation, or evaluation.
- The pooled encoder consumes channel-first `[B,C,T]`, preserves declared convolution order, and mean-pools after the final activation. It must not silently substitute the legacy dilated/BatchNorm architecture.
- Nested fitted values use generic runtime maps, not embedded static parameters: fusion maps by modality name, `Stages` maps by zero-based stage index encoded as canonical string keys, and `SupervisedObjective` exposes native-loss runtime parameters. Every layer delegates collision checks to `resolve_component`; provenance uses a separate fully materialized config.
- Packed fixed standardization is exactly `[values F, observed F] → [normalized F, unchanged observed F]` on the last axis. Per-instance observation emission is exactly `[normalized C, finite-observed C]` on channel axis 1. No alternative interleaving is permitted.

### Current architectures to reproduce

- Flat: observed Standardize; `Linear(116,h)→ReLU` tabular; `Linear(122,h)→ReLU` sensor; gate sensor; concatenate tabular, sensor, presence; `Linear(2h+1,h)→ReLU→Linear(h,4)`. `tabular` mode forces sensor/presence to zero without changing construction.
- Sequence: observed Standardize + `Linear(116,h)→GELU`; reshape signal `[B,T,8]`, derive finite observation, per-instance population standardize, concatenate normalized+observation, layout to `[B,16,T]`; two `Conv1d(k=9,s=4,p=4)` + GELU; mean time; gate; concatenate presence; `Linear(2h+1,h)→GELU→Linear(h,4)`.
- Flat loss is native weighted CrossEntropy mean. Sequence loss is `mean(group_weight * CrossEntropy(weight=class_weight, reduction="none"))`; sequence accuracy is weighted by the same sample weights.

### Scope boundaries

- Story 9.5 owns generic subgroup/ablation evaluation and deletion of both ablation helpers. Story 9.6 owns representative-tier MLflow parity. Do not pull either forward.
- Keep sequence bounded-memory streaming evaluation and window-to-participant aggregation consumer-local and catalogued; both are roadmap work.
- Do not add multi-field model inputs, a modality container/dataclass, registry/entry point, project dispatch, task/mode branches in DSio, weighted sampling, pretrained weights, GRU/LSTM/attention/TCN breadth, new dependencies, or task-spine changes.
- Preserve ingestion, packed feature meaning/order, split topology, standardization values, loss reductions, optimizer/trainer settings, flow topology, output field names, participant aggregation, submissions, goldens, dependencies, and lockfiles.
- DSio source and docstrings must not name either consumer; evidence remains outside `src/dsio`.

### Existing seams and files to preserve

- `src/dsio/experimental/model/compositions.py`: existing `MLP`, `Stages`, and `Chain`; `Stages` explicitly delegates branching to a named composition.
- `src/dsio/experimental/model/convolution.py`: `DenseConv1d`; legacy `components.Conv1dEncoder` is not the parity architecture.
- `src/dsio/experimental/model/layout.py`: checked `TimeMajorToChannelFirst`.
- `src/dsio/experimental/model/standardization.py`: fitted `Standardize` and per-instance `InstanceStandardize`; extend rather than duplicate their numeric guards.
- `src/dsio/experimental/model/objectives.py`: exact class/sample weighting, runtime loss parameter conversion, stateless metric contract.
- `src/dsio/experimental/inference/outputs/multiclass.py`: configure scores with field `probability` and pass the same output instance's `validator` to Predictor.
- `src/dsio/experimental/data/arrays.py`: reuse `collate_arrays`; do not create another array builder.

### Previous-story intelligence

- Story 9.1 established packed `WindowItems`, digest/order/group guarantees, and configured datasets.
- Story 9.2 established observed-only fitted/per-instance normalization and fit → record → runtime-inject provenance. It assigned pooled encoder, fusion, classifier/objective/output migration, and model factory closure to 9.4.
- Story 9.3 established fitted class/group weights, native/sample-weight reduction semantics, explicit neutral validation mappings, and collision protection. Its adversarial review caught identity, runtime injection, float32, and evidence-return gaps; test these boundaries rather than assuming them.
- Completed 9.3 baseline: flat 711, sequence 909, Kaggle total 5,569 non-ingestion lines, 11 banned definitions; full suite 1,370 passed, one explicit live test deselected, 188 warnings.

### Library and testing requirements

- Use the repository constraints: Python `>=3.12,<3.15`, PyTorch `>=2.7,<3`, Lightning `>=2.5,<3`, TorchMetrics `>=1.7,<2`, MLflow `>=3,<4`. No upgrade or dependency change is needed.
- Focused baseline before implementation: 178 relevant tests pass with 33 warnings. Golden file digest begins `71371a90` and must remain byte-unchanged.
- New tests belong with model compositions/signals/standardization plus both CMI portfolio suites. Include architecture-equivalence tests before expensive flow tests, component resolution, malformed configurations, absent-payload mutation, batch-dependent encoder isolation, output round-trip, provenance, and candidate/catalog assertions.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Delivery Rules; Epic 9 / Story 9.4]
- [Source: `_bmad-output/planning-artifacts/prd.md` — authority and precedence]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-1, CAP-10, constraints and non-goals]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — blocks #10, #15 and remaining local code]
- [Source: `_bmad-output/specs/spec-component-warehouse/conventions.md` — single `x`, layouts, configuration, catalog and evidence]
- [Source: `_bmad-output/specs/spec-component-warehouse/brownfield.md` and `roadmap.md` — reusable seams and deferrals]
- [Source: `docs/component-admission.md`]
- [Source: `_bmad-output/implementation-artifacts/9-1-map-windows-into-training-items-with-weights-and-groups.md`]
- [Source: `_bmad-output/implementation-artifacts/9-2-fit-observed-only-statistics-for-missing-data.md`]
- [Source: `_bmad-output/implementation-artifacts/9-3-balance-training-by-class-and-group.md`]
- [Source: both Child Mind `components.py`, training modules, and downstream/evaluation modules]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- Story created from the complete Warehouse epic, canonical SPEC companions, admission policy, current consumer/core code, candidate/evidence/catalog state, recent git history, and completed Stories 9.1-9.3.
- Required analysis was split across product/spec, code architecture, and test/governance reviews; findings were reconciled into the acceptance criteria and implementation slices above.
- Implementation plan: freeze seeded legacy equivalence first; implement generic fusion and flat migration; add exact pooled temporal composition and sequence migration; then close output/objective/array candidates, evidence, metrics, review, and full gates.

### Completion Notes List

- Added generic `MultimodalFusion`, `PooledConv1d`, packed/per-instance observation composition seams, nested runtime routing, and class-index accuracy with strict boundary validation.
- Migrated both Child Mind models, objectives, outputs, and flat array collation to configuration-only warehouse components while preserving seeded parameter/logit, loss, accuracy, golden, replay, export, and provenance behavior.
- Adversarial review resolved 12 patches; two pre-existing policy/design questions were recorded in `deferred-work.md`.
- Reduced non-ingestion lines from flat 711 to 684, sequence 909 to 906, and Kaggle 5,569 to 5,539; banned local definitions fell from 11 to 3.
- Verification: 1,390 passed and one explicit live test deselected; focused CMI/fusion tests, Ruff, formatting, mypy, catalog, build, eight distribution/consumer-flow contracts, and three import contracts all pass. Goldens, `pyproject.toml`, and `uv.lock` are unchanged.

### File List

- `_bmad-output/implementation-artifacts/9-4-compose-multimodal-fusion-with-modality-presence-gating.md`
- `_bmad-output/implementation-artifacts/deferred-work.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/evidence.yaml`
- `src/dsio/experimental/model/{__init__,compositions,convolution,fusion,objectives,standardization}.py`
- `src/dsio/experimental/inference/outputs/multiclass.py`
- `reference_projects/kaggle/child_mind/{components,tasks/downstream,tasks/training}.py`
- `reference_projects/kaggle/child_mind/sequence/{components,tasks/evaluation,tasks/training}.py`
- `tests/experimental/model/{test_compositions,test_fusion,test_objectives,test_signals}.py`
- `tests/experimental/inference/test_outputs.py`
- `tests/kaggle_portfolio/{test_child_mind,test_child_mind_sequence}.py`

## Change Log

- 2026-10-18: Created an implementation-ready story with exact fusion/temporal architecture, objective/output/array closure, seeded parity, evidence, and deletion contracts.
- 2026-10-18: Implemented and adversarially reviewed the fusion migration; all acceptance criteria and repository gates pass.
