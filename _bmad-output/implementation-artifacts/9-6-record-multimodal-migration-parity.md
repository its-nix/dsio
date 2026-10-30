# Story 9.6: Record Multimodal Migration Parity

Status: ready-for-dev

## Story

As a maintainer,
I want representative-tier parity for both Child Mind consumers recorded against the immutable baseline,
so that Epic 9 proves the warehouse preserved multimodal behavior on real data.

## Acceptance Criteria

1. Given the Story 6.6 baseline records for `child_mind` and `child_mind_sequence`, when parity preflight runs, then every referenced evaluation Run is `FINISHED`, its `provenance.json` records a clean tree at `fe37300c844de7db0c54189ec0c45b9c16cdd101`, its Python/Torch runtime and full-precision metrics equal the committed baseline record, and the current migration is a clean committed revision containing completed Stories 9.1–9.5. The existing flat fused/tabular and sequence baseline Runs currently fail the clean-provenance check; recover them from a clean historical worktree on their recorded hardware/runtime, reproduce every metric exactly, and replace only invalid Run identities/URIs and operational metadata. Baseline metric mappings, commit, and hardware remain unchanged.
2. Given valid clean baselines, when `child_mind` and `child_mind_sequence` rerun through the existing `tools/representative.py --compare --tolerance 0` path, then both use seed 19, isolated `PREFECT_HOME`, external workspaces, live MLflow, and their recorded runtime/hardware: Python 3.12.3 and Torch 2.13.0+cpu on Ryzen 9 9900X / CPU / 12 Torch threads for flat CMI; Python 3.12.3 and Torch 2.13.0+cu130 on RTX 5070 Ti / CUDA / 12 Torch threads for sequence CMI. The runner validates observed Python and Torch versions in addition to accelerator, device, and thread count before recording parity. Every full-precision metric name and value matches exactly at absolute tolerance zero: moved common metrics report signed deltas; missing/extra metrics report the available side with a null delta and explicit missing-metric evidence; non-finite metrics are rejected before a parity Run is created. Stories 9.1–9.5 declared no representative tolerance, so no post-observation tolerance is permitted.
3. Given a successful comparison, when MLflow evidence is inspected, then one dedicated parity Run per consumer identifies the baseline and observed evaluation Run IDs, commit IDs, consumer, hardware class, and tolerance `0.0`; logs `baseline.*`, `observed.*`, and signed `delta.*` values for every metric; is `FINISHED` and tagged `passed`; and has an immutable credential-free HTTPS tailnet URI. Flat CMI preserves its two evaluation identities in the parity tags. MLflow and the evidence register remain the result sources of truth; no second runner, result model, or committed parity-results hierarchy is added.
4. Given the two passing parity URIs, when component evidence is closed, then each URI is appended to every corresponding real-use record in `docs/component-warehouse/evidence.yaml`, existing Run URIs and `via` relationships remain intact, the CMI pair still counts as one unrelated use group, and no component is promoted automatically. The flat URI covers all 12 flat uses and the sequence URI covers all 20 sequence uses listed in Dev Notes. `dsio.eval.evaluate` receives no warehouse evidence entry because it is a stable spine function, and the sequence streaming evaluation candidate remains registered unchanged. The catalog regenerates and catalog/admission validation passes.
5. Given Epic 9's completed migrations, when deletion and repository gates run, then flat CMI remains 654 non-ingestion lines, sequence remains 906, their combined total remains 1,560, Kaggle remains 5,509, neither CMI consumer has a banned local definition, and the repository's only three banned definitions remain the explicit `self_supervised` fixtures. Relative to Story 6.6, the CMI pair is down 182 lines and Kaggle is down 285; Story 9.6 claims no release-wide 50% reduction. Both CMI contract/replay/export/provenance tiers, calibration and micro-batch invariance coverage, representative tooling, catalog/admission/import contracts, full pytest, Ruff, formatting, mypy, build, and distribution/consumer-flow contracts pass. Contract goldens, dependency declarations, lockfile, consumer computation, candidate boundaries, and baseline metric values remain unchanged.

## Tasks / Subtasks

- [ ] Task 1: Freeze the exact parity contract and clean-source gates (AC 1-2, 5).
  - [ ] Record hashes for `tests/golden_metrics.json`, `pyproject.toml`, `uv.lock`, and both baseline metric mappings before any representative work.
  - [ ] Require a clean committed revision containing Stories 9.1–9.5; never certify the current dirty worktree or hide changes with temporary stashing.
  - [ ] Preflight all three baseline evaluation Runs and retain their dirty-provenance failures as the reason historical recovery is required.
  - [ ] Add a failing representative-tool regression for observed Python/Torch mismatch, then extend the existing hardware preflight to reject runtime mismatch without adding a new runner or result type.
  - [ ] Keep datasets and run workspaces outside the repository; set the live tracking endpoint and credential-free tailnet UI base explicitly.
- [ ] Task 2: Recover the invalid historical CMI baseline identities (AC 1).
  - [ ] From an exclusive clean detached worktree at `fe37300c844de7db0c54189ec0c45b9c16cdd101`, rerun flat CMI on Python 3.12.3 / Torch 2.13.0+cpu / Ryzen 9900X / 12 threads.
  - [ ] Rerun sequence CMI on Python 3.12.3 / Torch 2.13.0+cu130 / RTX 5070 Ti / 12 threads without changing repository dependency files.
  - [ ] Prove every recovered metric equals the committed full-precision baseline exactly. Transplant each clean recovery record's complete `runs` mapping and update only `recorded`, `seconds`, `tracking_uri`, and credential-free `ui_uri` as needed; keep `commit`, `dirty: false`, `hardware`, and `metrics` identical, then regenerate the Markdown view.
  - [ ] Run `_require_baseline_runs` successfully for both recovered records before executing migrated consumers.
  - [ ] Obtain an owner-created clean checkpoint containing the recovered baseline pair, Story 9.6, runtime-preflight test/tool change, and completed Epic 9 migration. Run both current comparisons from that exact clean revision before editing evidence or catalog files.
- [ ] Task 3: Record exact current representative parity (AC 2-3).
  - [ ] Run `child_mind --compare --tolerance 0` from the clean migration revision on the recorded CPU class.
  - [ ] Run `child_mind_sequence --compare --tolerance 0` from the same revision in the CUDA environment; ensure editable/imported DSio points at that exact clean worktree.
  - [ ] Verify both parity Runs are `FINISHED`, tagged `passed`, tolerance `0.0`, and have maximum absolute delta `0.0`; verify flat tags preserve fused and tabular evaluation mappings.
  - [ ] If any metric moves, preserve the failed diagnostic Run and investigate reproducibility or a migration defect. Add a regression before any fix and return to exact parity; never invent a tolerance.
- [ ] Task 4: Attach evidence and record Epic 9 deletion (AC 4-5).
  - [ ] Append the flat parity URI to its 12 use records and the sequence URI to its 20 use records; preserve older calibration/telemetry evidence and fixture uses.
  - [ ] Regenerate and check the catalog; keep `dsio.eval.evaluate` outside warehouse evidence and retain the complete sequence streaming-evaluation candidate.
  - [ ] Run `tools/consumer_metrics.py` and record 654 / 906 / 1,560 / 5,509 with zero CMI and three repository banned definitions; do not rewrite the Story 6.6 deletion baseline.
  - [ ] Verify the generated baseline Markdown equals `render(parity-baseline.json)` and every added URI belongs to the matching consumer's passing parity Run.
- [ ] Task 5: Close Epic 9 with complete verification (AC 1-5).
  - [ ] Run both CMI portfolio suites, representative tests, component catalog tests, admission coverage, sequence calibration/micro-batch invariance, and unchanged contract goldens.
  - [ ] Run full pytest, Ruff lint/format, mypy, all three import contracts, catalog generation/check, build, and distribution/consumer-flow contracts after restoring the standard CPU environment.
  - [ ] Confirm `tests/golden_metrics.json`, `pyproject.toml`, `uv.lock`, baseline metric mappings, consumer computation, and sequence candidate content are unchanged except for authorized clean baseline identities/evidence URIs.
  - [ ] Run the three-layer adversarial review, resolve every actionable finding, then mark Story 9.6 and Epic 9 `done`.

## Dev Notes

### Exact representative metrics

Binding values come from `docs/component-warehouse/parity-baseline.json`, never the rounded Markdown rendering.

- Flat fused: accuracy `0.4824561403508772`; QWK `0.36875764431263147`; sensor present `0.3610938899221602`; missing `0.3736574526110231`; ablated `0.3333431525613456`; delta `0.02775073736081457`.
- Flat tabular: accuracy `0.4649122807017544`; QWK `0.3596520893661095`; sensor present `0.32445734477536603`; missing `0.3782564636892982`; ablated `0.32445734477536603`; delta `0.0`.
- Sequence: accuracy `0.5519911504424779`; QWK `0.3792779673786537`; sensor present `0.37762161335096556`; missing `0.38014027720307375`; ablated `0.32920402740626853`; delta `0.04841758594469703`.

### Evidence coverage

- Flat, 12 real-use records: `StoredItems`; `fit_class_weights`; `fit_standardization`; `record_fitted`; `Stages`; `SupervisedObjective`; `MulticlassOutput`; `MulticlassValidator` via `MulticlassOutput`; `collate_arrays`; `MultimodalFusion`; `PackedObservedStandardize`; `ClassIndexAccuracy`.
- Sequence, 20 real-use records: `WindowItems`; `calibrate_execution`; `log_phase_evidence`; `measure_phase`; `calibrate_training_execution`; `log_calibration`; `fit_class_weights`; `fit_group_weights`; `fit_standardization`; `record_fitted`; `Stages`; `TimeMajorToChannelFirst`; `InstanceStandardize`; `SupervisedObjective`; `MulticlassOutput`; `MulticlassValidator` via `MulticlassOutput`; `MultimodalFusion`; `PooledConv1d`; `PackedObservedStandardize`; `ClassIndexAccuracy`.
- Append parity evidence to existing `runs` lists. Do not erase the sequence calibration/telemetry Runs already present. Do not add evidence for consumer-local functions or the stable `evaluate` spine.

### Operational boundaries

- Reuse `tools/representative.py`; it already provides subprocess isolation, baseline preflight, hardware-class checks, exact comparison, MLflow parity Runs, and immutable UI URIs. Do not create another CLI or result schema.
- The current checkout is not certifiable: `HEAD` is `9727fa3`, while Stories 9.1–9.5 are uncommitted. `record_parity` rejects dirty observed sources. Obtain an owner-created clean commit before representative execution; this story does not authorize committing unrelated work.
- Current baseline evaluation IDs `dbc483ed18b046a29acad31931b7c131`, `054ff4f568c24b42a0be91265269c0f5`, and `9eb0fe3346f244908747ebcaf1ee4434` have dirty MLflow provenance and must not be reused as valid baselines. Recovery modifies committed baseline files, so the initial clean Epic 9 commit is insufficient: create a second owner-authorized clean checkpoint after recovery/tool hardening and before current comparisons.
- The standard `.venv` is Torch 2.13.0+cpu. Sequence recovery/current runs require a separate or temporary Python 3.12 CUDA 2.13.0+cu130 environment on the RTX 5070 Ti. Do not edit `pyproject.toml` or `uv.lock`, and do not let `uv sync` restore CPU Torch during the CUDA run. Restore the standard CPU environment before repository gates.
- Use `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`, `MLFLOW_UI_URI=https://pop.tailee691f.ts.net:8443`, an isolated `PREFECT_HOME`, and an external Story 9.6 runs root. Never commit credentials, local file-store URIs, datasets, workspaces, or generated model artifacts.
- If clean source, real data, live tracking/UI, exact runtime, or recorded hardware is unavailable, leave the story in progress and report the exact prerequisite. Local contract/golden tests are readiness evidence, never representative parity evidence.

### Scope boundaries and preservation

- This is an evidence/governance closure story, not another component migration. Expected repository changes are this story, sprint status, recovered baseline JSON/Markdown, evidence YAML, regenerated catalog, and the narrow observed-runtime preflight/test in `tools/representative.py` and `tests/test_representative.py`.
- No expected changes to consumer/core code, candidates, contract goldens, dependencies, or lockfile. Change other code only if a zero-tolerance run exposes a real migration defect; write the failing regression first.
- Do not alter models, initialization, standardization, fitted statistics, weighting, loss reduction, evaluation polarity, split/flow topology, output schemas, submissions, bounded-memory participant aggregation, subgroup scoring, or sequence ablation.
- Do not pull Epic 10 fit/export/downstream spine work forward. Do not promote components; Story 11.1 owns promotion. The CMI pair remains one unrelated use.
- Failed and diagnostic MLflow Runs remain immutable but are not attached as passing evidence.

### Previous-story intelligence

- Story 9.4 preserved exact flat and sequence model behavior while replacing both local classifiers with `MultimodalFusion`; it closed at 1,390 tests and left three unrelated fixture findings.
- Story 9.5 migrated flat CMI to one `evaluate` call, retained sequence streaming evaluation as a roadmap candidate, fixed delta polarity to full-minus-ablated, and closed at 1,402 tests before review hardening. Its final measures are the Story 9.6 expected no-change values: flat 654, sequence 906, Kaggle 5,509, banned 3.
- Story 8.7 established the binding dirty-baseline recovery precedent: rerun the clean historical commit on matching hardware/runtime, reproduce metrics exactly, and replace only invalid Run identities. It also forbids post-observation tolerance.
- Current focused readiness evidence is green: 136 representative/consumer-metrics/catalog/admission/CMI tests and catalog check passed. These are not substitutes for real representative Runs.

### Project structure and verification

- Expected files: `_bmad-output/implementation-artifacts/9-6-record-multimodal-migration-parity.md`, `_bmad-output/implementation-artifacts/sprint-status.yaml`, `tools/representative.py`, `tests/test_representative.py`, `docs/component-warehouse/parity-baseline.json`, `docs/component-warehouse/parity-baseline.md`, `docs/component-warehouse/evidence.yaml`, and generated `docs/component-warehouse/catalog.md`.
- Standard focused gate: both CMI portfolio suites, `tests/test_representative.py`, `tests/test_consumer_metrics.py`, `tests/test_component_catalog.py`, and `tests/experimental/test_admission_coverage.py`.
- Full gate: default pytest with only the explicit live test deselected; Ruff check/format; mypy; `lint-imports`; catalog generation/check; `uv build`; distribution and consumer-flow contracts.
- Preserve `tests/golden_metrics.json` SHA-256 `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530` and record dependency/lockfile hashes before execution.

### References

- [Source: `_bmad-output/planning-artifacts/prd.md` — authority and precedence]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-10 and constraints]
- [Source: `_bmad-output/specs/spec-component-warehouse/conventions.md` — evidence register]
- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Story 9.6 and delivery rules]
- [Source: `_bmad-output/implementation-artifacts/8-7-record-sequence-migration-parity.md` — dirty-baseline recovery precedent]
- [Source: `_bmad-output/implementation-artifacts/9-5-evaluate-subgroups-and-modality-ablation.md` — completed migration and preservation boundary]
- [Source: `tools/representative.py` and `tests/test_representative.py`]
- [Source: `docs/component-warehouse/parity-baseline.json`]
- [Source: `docs/component-warehouse/evidence.yaml`]
- [Source: `docs/component-warehouse/deletion-baseline.md`]
- [Source: `docs/component-admission.md`]

## Dev Agent Record

### Agent Model Used

GPT-5

### Implementation Plan

- Establish clean committed migration and recover clean historical CMI baselines before any current-code representative run.
- Compare flat and sequence consumers at zero tolerance on their recorded CPU/CUDA hardware classes.
- Attach passing parity URIs per callable, regenerate governance artifacts, measure deletion, and close all gates.

### Debug Log References

- Creation analysis confirmed all three CMI baseline evaluation Runs have dirty MLflow provenance despite `parity-baseline.json` recording `dirty: false`.
- Creation analysis confirmed the data, live MLflow/UI, CPU, and RTX 5070 Ti are available; the current repository environment is CPU-only and the migration tree is uncommitted.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Independent checklist validation applied clean-checkpoint, observed-runtime, diagnostic, and historical-recovery safeguards.

### File List

- `_bmad-output/implementation-artifacts/9-6-record-multimodal-migration-parity.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

## Change Log

- 2026-10-19: Created implementation-ready Story 9.6 with clean-baseline recovery and zero-tolerance parity gates.
