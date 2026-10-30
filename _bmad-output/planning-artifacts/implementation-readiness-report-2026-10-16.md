---
stepsCompleted:
  - step-01-document-discovery
  - step-02-prd-analysis
  - step-03-epic-coverage-validation
  - step-04-ux-alignment
  - step-05-epic-quality-review
  - step-06-final-assessment
filesIncluded:
  - README.md
  - CONTEXT.md
  - docs/superpowers/specs/2026-08-15-dsio-design.md
  - docs/superpowers/specs/2026-08-20-dsio-lean-design.md
  - docs/superpowers/specs/2026-09-18-generic-experiment-spine.md
  - _bmad-output/planning-artifacts/epics.md
  - _bmad-output/planning-artifacts/epics-component-warehouse.md
reviewEvidence:
  - architecture_review.md
---

# Implementation Readiness Assessment Report

**Date:** 2026-10-16
**Project:** dsio

## Document Inventory

A conventionally named PRD was not present during initial discovery. The follow-up course-correction review found `_bmad-output/specs/spec-component-warehouse/SPEC.md`, which explicitly declares itself and its listed companions to be the canonical Warehouse v1 contract. `_bmad-output/planning-artifacts/prd.md` now provides a discoverable authority index. The accepted generic-spine specification governs the completed 0.2 foundation; the Warehouse SPEC and companions govern Warehouse v1/0.3. Both epic documents are included because they cover those separate scopes. `README.md` documents shipped behavior but is not requirements authority. `architecture_review.md` is supporting review evidence. UX documentation is non-applicable because the sources define a Python library with no UI deliverable.

## PRD Analysis

### Source precedence

The accepted `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` governs the 0.2 architecture and explicitly supersedes the 2026-08-20 lean design, which supersedes the 2026-08-15 design. The two earlier designs are implementation history. For Warehouse v1/0.3, `_bmad-output/specs/spec-component-warehouse/SPEC.md` and every companion in its frontmatter are the complete canonical contract; that SPEC explicitly supersedes earlier portfolio guardrails where stated. `CONTEXT.md` fixes domain vocabulary. `README.md` describes shipped behavior without overriding either contract.

### Functional Requirements

FR1: DSio shall be an installable, versioned Python package consumed by independent projects through pinned dependency lockfiles rather than cloning, vendoring, or merging DSio source.

FR2: Consumer projects shall own their Prefect flow topology and typed flow configuration; DSio shall not provide a parallel DAG model, orchestration protocol, runner, scheduler, deployment layer, or project workflow CLI.

FR3: DSio shall resolve or create a native MLflow Experiment without creating a Run.

FR4: Every evidence-producing Prefect task attempt shall create one visible top-level native MLflow Run and explicitly record its successful, failed, or cancelled terminal status.

FR5: Attempt Runs shall be tagged with the associated Prefect flow-run, task-run, dynamic-key, retry/attempt, and stable Execution Identity information needed to relate retries without rewriting failed history.

FR6: DSio shall calculate Execution Identity from every relevant input capable of changing an Experiment Node's output, including normalized inputs/configuration, dataset and split identity, randomness, component identity, consumer code and dirty patch, lockfile, DSio package identity, command/environment, and topology when materially relevant.

FR7: DSio shall record safe normalized configuration and reproducibility provenance as native MLflow evidence while excluding declared credentials, secrets, and ephemeral fields.

FR8: DSio shall resolve reusable evidence only from a verified successful native MLflow Run whose immutable identity, provenance, and caller-declared required artifacts agree; missing or invalid evidence shall be a cache miss.

FR9: DSio shall produce and consume immutable `runs:/<run-id>/<artifact>` evidence references; mutable aliases or stages shall not satisfy exact evidence resolution.

FR10: Pure serializable Prefect tasks may use DSio's deterministic cache-key utility, while MLflow-producing tasks shall validate MLflow evidence instead of treating a cached identifier as proof.

FR11: A downstream-only rerun shall be possible by validating prior source Runs and passing exact evidence URIs to the project-selected downstream tasks without DSio inferring a rerun plan or invoking upstream work.

FR12: DSio shall expose one stable canonical storage API backed by the single selected format and shall not expose a public backend registry.

FR13: The selected store format shall support building, reopening, stable identity and positional lookup, bounded metadata validation, per-sample payload digest validation, and an explicit full-store integrity scan.

FR14: DSio shall support lazy identity-preserving sample/window access so corpora are stored once and cheap/combinatorial views are indexed rather than copied.

FR15: DSio shall generate splits through one closed DSio-owned dispatcher; projects shall not register splitters at runtime.

FR16: Split generation shall delegate established algorithms where appropriate and add deterministic identity, validation, and serialization.

FR17: A Split Manifest shall bind assignments to an immutable Dataset Identity and record algorithm identity and parameters, seed, named partitions/roles, group evidence, dependency provenance, validation results, membership evidence, and a digest.

FR18: Split evidence shall be recordable and reloadable through native MLflow dataset inputs and immutable artifacts without copying source data.

FR19: Split validation shall prevent leakage, including mutually disjoint membership and the documented group/row overlap guarantees.

FR20: `DsioDataModule` shall consume the store, exact example identities, Split Manifest, role mapping, and named dataset factory; construct datasets in `setup(stage)`; and exclusively own deterministic DataLoader construction.

FR21: Every item and batch shall preserve exact ordered `sample_id` values, enforce batch-field cardinality, and keep CPU collation on CPU.

FR22: `DsioModule` and `DsioDataModule` shall be the only supported training composition roots, instantiated directly and protected against subclassing.

FR23: `DsioModule` shall accept a model, objective callable, native optimizer/scheduler factory configuration, metrics/callbacks, and prediction behavior without dispatching on project or task names.

FR24: The objective shall receive `(model, batch, stage)` and return a flat mapping containing a mandatory scalar tensor `loss` and optional named scalar tensor metrics logged through native Lightning behavior.

FR25: Reusable components shall be named, importable units resolved from explicit reference-and-parameters mappings; lambdas, closures, local definitions, and `__main__` objects shall be rejected.

FR26: Component mappings shall be validated before execution and recorded in Execution Identity/provenance without introducing a runtime component registry or DSio-specific optimizer/scheduler wrappers.

FR27: Stochastic train-only accelerator augmentation shall execute exclusively in `DsioModule.training_step()` after transfer and before the objective; deterministic preprocessing shared with inference shall belong to the Predictor; CPU decoding, windowing, batching, and collation shall belong to data components.

FR28: Augmentation randomness shall derive from stable execution seed, epoch, sample or ordered batch identity, view identity, and augmentation identity so retries and checkpoint resumes reproduce unchanged executions.

FR29: Unsupported augmentation device or dtype combinations shall fail preflight rather than silently falling back.

FR30: Experiment-level evaluation shall run as an ordinary Prefect node using native MLflow evaluation plus DSio-owned native metrics only where the ecosystem lacks them.

FR31: Evaluation shall support schema-valid, identity-preserving outputs, including masked dense evaluation with explicit shape checks and per-target scoring where declared.

FR32: The MLflow Model Signature shall be the canonical Predictor interface, supplemented by named DSio validation for task-family constraints MLflow does not enforce.

FR33: DSio shall export an MLflow Logged Model using the native PyTorch flavor when suitable or PyFunc when deterministic preprocessing/structured output requires it; export is required only for nodes declaring Predictor output.

FR34: `dsio.inference.predict(model_uri, inputs)` shall load through MLflow PyFunc, invoke prediction, preserve `sample_id`, and validate declared output.

FR35: A resumable Lightning checkpoint shall contain the full Training System, while an exported Predictor shall contain only deterministic preprocessing, prediction behavior, signature/example, and predictor state.

FR36: DSio shall provide supervised and self-supervised synthetic reference flows demonstrating canonical storage, split generation, exact training classes, augmentation boundaries, MLflow evidence/evaluation/export, failed-attempt visibility, retry, verified reuse, downstream-only reevaluation, and schema-validated inference.

FR37: DSio shall publish a generated component catalog containing public component batch contracts, maturity, and supporting consumer evidence, with CI rejecting catalog drift.

FR38: Component maturity shall be represented by package location and governed by the admission policy: one real downstream use for Experimental status and an unrelated second use, compatibility tests, migration discipline, and human approval for Stable status.

FR39: DSio shall support fixed-size items through the canonical store/windowing model, with an explicit one-window-per-entity assertion preventing variable-size items from silently becoming crops.

FR40: DSio shall not implement promotion policy, deployment, project lifecycle, remote scheduling, project-specific label semantics, serving strategy, infrastructure provisioning, a backend plugin system, universal trainer, parallel tracking model, or project-specific CLI.

Total FRs: 40

### Non-Functional Requirements

NFR1: Reproducibility — unchanged Execution Identity, including retries and checkpoint resumes, shall reproduce deterministic data, augmentation, training inputs, and reusable evidence.

NFR2: Integrity — artifact, sample, store, split, and model evidence shall be content-identified and validation shall fail closed on mismatches rather than return unverified data.

NFR3: Reliability — no task or flow failure shall be converted into partial success; required lifecycle-evidence write failures shall fail the Attempt.

NFR4: Auditability — failed Attempts shall remain visible and immutable, and lineage shall use native MLflow inputs and immutable source references.

NFR5: Concurrency safety — tracking shall target explicit Run IDs and shall not rely on MLflow ambient active-Run state inside concurrent tasks.

NFR6: Secret safety — credentials shall never be logged, while non-secret resource identity needed for reproduction shall remain explicit.

NFR7: Compatibility — Stable APIs shall follow semantic versioning and require migration notes for breaking changes; Experimental APIs may change between minor releases.

NFR8: Portability — supported Components shall be importable in another process and consumer projects shall pin immutable DSio releases or commits.

NFR9: Native interoperability — Prefect, Lightning, PyTorch/TorchMetrics, and MLflow entities and lifecycle behavior shall remain native rather than wrapped by competing DSio abstractions.

NFR10: Testability — component admission requires relevant integration or property tests, deterministic behavior, documentation, provenance, and adversarial review; Stable admission additionally requires compatibility tests.

NFR11: Performance/storage efficiency — corpora shall be stored once, cheap/combinatorial views shall remain indexes, and materialization shall be limited to expensive deterministic work.

NFR12: Data correctness — sample identity and ordering shall survive item, batch, prediction, evaluation, and evidence boundaries.

NFR13: Leakage resistance — split/group/row invariants and train/inference boundaries shall be enforced structurally and tested, not left only to review convention.

NFR14: Usability — a new consumer project shall be able to install a pinned DSio release, define an ordinary Prefect flow, and use the documented public interfaces without adopting a DSio lifecycle or repository template.

NFR15: Maintainability — top-level packages shall represent cohesive pipeline domains; modules become packages only when distinct internal responsibilities justify the additional surface, with public imports preserved through re-exports.

NFR16: Scope discipline — support claims shall cover only environments exercised by the published compatibility matrix.

NFR17: Scientific validity — validation reports scientific acceptance separately from promotion/deployment decisions, and evaluation evidence must remain tied to immutable dataset/model/run identity.

NFR18: Error clarity — unsupported configurations, unsafe names, shape mismatches, invalid masks, non-importable components, missing evidence, and integrity failures shall be rejected before misleading evidence is written.

Total NFRs: 18

### Additional Requirements and Constraints

- Python, Prefect, Lightning, PyTorch, MLflow, and their native contracts are required architectural dependencies.
- The canonical storage choice must be justified by a focused benchmark covering ingestion, contiguous/random access, concurrent readers, memory, disk size, interrupted-write recovery, synthetic workloads, and at least one representative real corpus.
- Project-specific Components remain in consumer projects until they meet DSio admission criteria.
- Batch inference is supported through immutable Predictor evidence; online serving and deployment remain out of scope.
- `CONTEXT.md` establishes canonical domain terms and explicitly deprecated synonyms; implementation and documentation should use that vocabulary consistently.
- Historical specifications contain many requirements that were explicitly superseded, including repository cloning/merging, a DSio project CLI, fixed fold-as-process orchestration, local-infrastructure ownership, and older tracking abstractions. They must not be used as current readiness criteria.

### PRD Completeness Assessment

The combined authority is complete enough for implementation. The generic-spine specification defines the 0.2 architecture and boundaries; the preservation-validated Warehouse SPEC supplies stable capability IDs, explicit release scope, constraints, non-goals, measurable success signals, and normative companions for 0.3. The requirements are technical rather than persona-oriented and do not define quantitative service objectives or a security threat model, but those omissions do not make Warehouse scope ambiguous. README features such as fixed-size items and masked dense evaluation are shipped 0.2 behavior and test evidence, not additional Warehouse-v1 requirements.

## Epic Coverage Validation

The core epic document contains 38 internally numbered FRs derived directly from the accepted generic-spine design and maps all 38 to Epics 1–5. The Warehouse epic document adds 18 separately numbered FRs, of which FR5 is explicitly withdrawn, and maps all 17 active Warehouse FRs to Epics 6–11. The 40-item matrix below is retained as an expanded shipped-surface cross-check assembled before the canonical Warehouse SPEC was discovered; it is not the authoritative Warehouse coverage calculation.

### Coverage Matrix

| FR | Requirement summary | Epic/story coverage | Status |
|---|---|---|---|
| FR1 | Installable pinned package | E1/S1.1 | Covered |
| FR2 | Project-owned Prefect orchestration; no DSio DAG/CLI | E1/S1.1, S1.2, S1.6 | Covered |
| FR3 | Resolve MLflow Experiment without Run | E1/S1.2 | Covered |
| FR4 | Visible Run per evidence-producing Attempt and terminal status | E1/S1.3 | Covered |
| FR5 | Prefect/retry/Execution Identity tags without rewritten history | E1/S1.3 | Covered |
| FR6 | Execution Identity includes all material inputs | E1/S1.4 | Covered |
| FR7 | Safe normalized configuration and provenance | E1/S1.4 | Covered |
| FR8 | Reuse only verified successful evidence with required outputs | E1/S1.5 | Covered |
| FR9 | Exact immutable evidence URIs; reject mutable references | E1/S1.5 | Covered |
| FR10 | Prefect caching for pure work; MLflow validation for evidence | E1/S1.5 | Covered |
| FR11 | Downstream-only reruns from immutable evidence | E1/S1.6 | Covered |
| FR12 | One canonical store and no public backend registry | E2/S2.1, S2.2 | Covered |
| FR13 | Build/reopen/lookup plus bounded metadata, sample digest, full verify | E2/S2.2 plus shipped 0.2 tests | Shipped; outside Warehouse scope |
| FR14 | Store once and lazy identity-preserving samples/windows | E2/S2.2, S2.5; Warehouse S7.1, S9.1 | Covered |
| FR15 | Closed split dispatcher | E2/S2.3 | Covered |
| FR16 | Delegated split algorithms plus DSio identity/validation/serialization | E2/S2.3 | Covered |
| FR17 | Complete immutable Split Manifest | E2/S2.3 | Covered |
| FR18 | Native MLflow split inputs/artifacts and reload | E2/S2.4 | Covered |
| FR19 | Split leakage and overlap validation | E2/S2.3 | Covered |
| FR20 | DataModule consumes manifest/roles/factory and owns loaders | E2/S2.5 | Covered |
| FR21 | Ordered sample identity, cardinality, CPU collation | E2/S2.5; Warehouse S6.3, S7.6, S8.1 | Covered |
| FR22 | Exact non-subclassable DSio training classes | E3/S3.1 | Covered |
| FR23 | Native injected training components without task dispatch | E3/S3.1, S3.2; Warehouse S10.1 | Covered |
| FR24 | Objective contract with scalar loss/metrics | E3/S3.1; Warehouse S7.4, S8.2 | Covered |
| FR25 | Named importable components; reject lambdas/closures/locals | E3/S3.2 | Covered |
| FR26 | Validate/record component mapping; no parallel wrappers | E3/S3.2 | Covered |
| FR27 | Exclusive augmentation/preprocessing/data ownership lanes | E3/S3.3; E4/S4.1 | Covered |
| FR28 | Reproducible augmentation identity | E3/S3.3 | Covered |
| FR29 | Preflight unsupported device/dtype without fallback | E3/S3.4 | Covered |
| FR30 | Evaluation as Prefect node using native evidence | E4/S4.4 | Covered |
| FR31 | Schema-valid evaluation including masked dense semantics | E4/S4.4 plus shipped 0.2 tests | Shipped; outside Warehouse scope |
| FR32 | MLflow Signature plus semantic Predictor validation | E4/S4.2 | Covered |
| FR33 | Native PyTorch/PyFunc export when declared | E4/S4.2; Warehouse S10.4 | Covered |
| FR34 | Immutable-URI inference preserving identity | E4/S4.3 | Covered |
| FR35 | Checkpoint/Logged Model responsibility separation | E3/S3.1; E4/S4.1 | Covered |
| FR36 | Supervised and SSL acceptance flows | E5/S5.2, S5.3 | Covered |
| FR37 | Generated evidence-backed component catalog with CI drift check | E6/S6.4; E11/S11.3 | Covered |
| FR38 | Evidence-based Experimental/Stable admission | E5/S5.1; E6/S6.5; E11/S11.1 | Covered |
| FR39 | Fixed-size item path with one-window-per-entity guard | Shipped 0.2 behavior and dedicated tests | Shipped; outside Warehouse scope |
| FR40 | Explicit non-goals and ownership boundaries | Epics 1–5 additional requirements and boundary acceptance criteria; Warehouse delivery rules | Covered |

### Scope Reconciliation

The initial review incorrectly promoted three README-described 0.2 behaviors—full-store verification, masked dense evaluation, and fixed-size whole-item safety—into additional Warehouse requirements. They are implemented and tested shipped behavior, but the canonical Warehouse SPEC does not include them in its v1 capability cohort. No new Warehouse stories are required for those behaviors.

### Coverage Statistics

- Generic-spine requirements: 38/38 mapped to Epics 1–5.
- Active Warehouse requirements: 17/17 mapped to Epics 6–11.
- Withdrawn Warehouse requirements: 1 (Warehouse FR5), deliberately excluded.
- Canonical planning coverage: 100%.
- Delivery coverage is not complete: Epics 7–8 are complete on `main`; Epics 9–11 remain backlog.

## UX Alignment Assessment

### UX Document Status

No UX document exists. This is appropriate: the accepted design and both epic documents define DSio as a Python library and explicitly state that UX/UI is not applicable. The sole UI reference is MLflow's externally provided local UI, which DSio neither designs nor owns.

### Alignment Issues

No product UI, web, mobile, dashboard, or frontend deliverable is implied. Developer experience is the relevant usability surface and is already represented by installability, native framework contracts, public API examples, actionable errors, a generated component catalog, and an external wheel-only acceptance consumer.

### Warnings

No missing-UX warning is warranted. However, because developer experience substitutes for conventional UX here, API discoverability and onboarding evidence should be treated as product acceptance criteria rather than documentation polish; Warehouse Stories 6.4 and 11.2 provide the strongest planned proof.

## Epic Quality Review

### Overall structure

Epics 1–5 are well formed: their titles and goals describe consumer outcomes, their dependency order is coherent, and their stories progress from installation/tracking through data, training, inference, and governed reuse. Acceptance criteria consistently use testable Given/When/Then clauses, include failure cases, preserve native-system boundaries, and maintain explicit FR traceability.

Epics 6–11 also describe consumer value rather than infrastructure for its own sake, and they correctly identify the work as brownfield migration. Their delivery rules provide unusually strong parity, test, evidence, deletion, and human-approval gates. The remaining execution concern is story size; the approved course correction adds mandatory vertical implementation slices without changing requirement IDs.

### Historical sequencing note

Story 6.4 depended on Story 6.5 and was implemented after it despite the numeric order. Both stories and Epic 6 are now complete. Renumbering completed artifacts would create traceability churn without improving future execution, so their IDs are deliberately preserved.

### Major issues

1. **Several future warehouse stories are migration programs rather than independently estimable changes.** Stories spanning multiple consumers or component families have independent failure and review surfaces.
   - Resolution: the delivery rules now require each large story to declare coherent component-to-consumer vertical slices, keep all gates green, and record parity/evidence/deletion deltas per slice while retaining the parent requirement ID.

2. **Story 10.1 originally made its ADR appear to approve its own architecture.** The canonical Warehouse SPEC already accepts CAP-7 with precise constraints.
   - Resolution: Story 10.1 now requires the ADR to record the accepted CAP-7 distinction and amend the generic-spine specification before production implementation is merged.

3. **Historical objective wording is imprecise.** Core epic FR14/S3.1 says the objective consumes “the batch and model output,” while the accepted generic-spine and Warehouse contracts—and the implementation—use `(model, batch, stage) -> mapping`.
   - Assessment: the executable contract and current normative specifications agree. The completed epic wording is retained as history; future stories must use the canonical signature.

### Minor concerns

1. Epic numbering continues correctly from 5 to 6, but both documents restart FR numbering. Cross-document references such as “FR14” are therefore ambiguous without a namespace.
   - Remediation: use `SPINE-FR-*` and `WAREHOUSE-FR-*`, or capability IDs, in all coverage and implementation artifacts.

2. Many warehouse criteria name exact consumer files and deletion targets, which is excellent for verification but brittle under harmless code movement.
   - Remediation: retain the semantic/syntax-tree checks as authoritative and treat filenames as discovery hints rather than the only pass condition.

3. The core epic document says all work is synthetic and CI-fast, while the warehouse scope adds live representative datasets and MLflow runs. The tier distinction is documented, but execution environment, ownership, and failure policy for representative tiers are not centralized.
   - Remediation: publish one test-tier policy distinguishing required CI gates from manually or periodically run evidence gates.

### Best-practice checklist

- User-value epics: Pass for all 11 epics.
- Epic dependency direction: Pass.
- Within-epic story direction: Historical S6.4/S6.5 irregularity; no future blocker.
- Story sizing: Controlled by the new mandatory vertical-slicing delivery rule.
- Acceptance criteria clarity and error coverage: Strong pass overall.
- Database/entity timing: Not applicable; native MLflow/Postgres infrastructure is outside current DSio ownership.
- Starter template requirement: Not applicable; this is explicitly a brownfield versioned library.
- Migration/compatibility planning: Strong pass with per-slice parity, evidence, deletion, and green-gate requirements.
- FR traceability: Complete for both canonical scopes; duplicate local FR numbering remains a minor readability concern, mitigated by Warehouse capability IDs.

## Repository Implementation Evidence

### Current delivery state

- The repository was reconciled to branch `main` at commit `9727fa3`; the package remains version `0.2.0` pending the Warehouse release.
- Implementation artifacts and sprint status mark every story in Epics 1–8 as `done`. Epics 7 and 8 provide the tabular and sequence Warehouse migrations that were absent from the initially assessed Story 6.6 branch.
- Epics 9–11 remain backlog. Multimodal migration, task-function consolidation, stable promotions, the external warehouse-first consumer, and the 0.3.0 release are not complete.
- The immutable Story 6.6 baseline remains 5,794 Kaggle non-ingestion lines and 70 banned local definitions. A current measurement on 2026-10-17 reports 5,673 Kaggle non-ingestion lines and 13 banned definitions: 10 registered Epic 9 Child Mind candidates and 3 explicit self-supervised fixtures.
- Representative parity is complete for the tabular and sequence cohorts. Story 8.7 records exact zero-tolerance parity for Essay Scoring, Parkinson FoG, and ROGII and leaves the golden metrics unchanged.

### Verification performed on 2026-10-16

- `uv build`: passed; produced the 0.2.0 wheel and source distribution.
- `uv run pytest -q`: 1,071 passed, 1 deselected, 171 warnings in 397.20 seconds.
- The deselected test is the explicitly live compose-stack round trip: `tests/train/test_artifacts.py::test_live_round_trip_against_the_compose_stack`. The compose stack was not running, so this operational check was not executed.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed for 341 files.
- `uv run mypy`: passed for 108 source files.
- `uv run lint-imports`: all three contracts kept across 152 files and 575 dependencies.
- The tests include installed-wheel consumer isolation, supervised and self-supervised reference flows, nine Kaggle contract flows, fixed-size-item behavior, masked evaluation, tracking/reuse, store/split integrity, training, export, and inference.

### Main-branch reconciliation evidence

- `git rev-list --left-right --count feat/6-6-baselines...main` reported `4 26`; the side branch used by the initial assessment predated the merged Warehouse work.
- Story 7.1 was merged as `fcce9f6`; Epics 7 and 8 were subsequently completed through Story 8.7 and are marked `done` in the main-branch sprint status.
- Story 8.7 records a passing build, 1,327 main tests with one live test deselected, Ruff lint/format, mypy over 124 source files, all three import contracts, catalog/admission checks, unchanged goldens, and exact sequence parity.
- `uv run python tools/consumer_metrics.py` was rerun on 2026-10-17 and produced the current 5,673-line / 13-definition measurement above.

### Verification caveats

- The default suite emits 171 warnings. Most are expected Lightning DataLoader/test-fixture warnings, but recurring MLflow PyFunc signature-inference warnings and upstream deprecations should be triaged before a major compatibility claim.
- The full gate was not rerun during the 2026-10-17 documentation reconciliation; the newest complete evidence is Story 8.7's recorded main-branch gate.
- Live MLflow/Postgres round-trip verification remains outstanding for this assessment.

## Summary and Recommendations

### Overall Readiness Status

**NEEDS WORK for the full current vision; READY for the narrower DSio 0.2 generic experiment spine.**

The accepted 2026-09-18 spine is substantially implemented and has strong automated evidence: package build, 1,071 default tests, type/lint/format gates, structural import boundaries, isolated-wheel consumers, two generic reference flows, and nine Kaggle contract flows. It is credible as a 0.2 reusable experiment spine.

It is not ready to claim the complete Component Warehouse v1/0.3.0 purpose defined by the canonical Warehouse SPEC and its companions. The trust foundation and the tabular and sequence cohorts (Epics 6–8) are complete. Epics 9–11 still contain the multimodal cohort, task-spine consolidation, stable promotion, external-consumer proof, and release evidence needed to make the full “warehouse of battle-tested blocks” claim.

### Critical Issues Requiring Immediate Action

1. **Do not declare Warehouse v1 ready before Epics 9–11.** Current measurement has reduced banned definitions from 70 to 13, but the planned 50% non-ingestion-line reduction, multimodal migration, stable promotions, wheel-only external consumer, and 0.3.0 release proof are still absent.
2. **Apply the large-story slicing rule during execution.** Each multi-consumer or multi-component story must land through green, evidence-backed vertical slices while the parent story remains in progress.
3. **Record CAP-7 architecture before implementing Story 10.1.** The ADR and generic-spine amendment must precede production fit-function code.
4. **Run the operational gate.** Start the documented compose stack and execute the one live MLflow/Postgres artifact round-trip before calling 0.2 operationally verified in a fresh environment.
5. **Assign warning ownership.** Triage recurring MLflow signature-inference warnings and upstream deprecations before making broad compatibility claims.

### Recommended Next Steps

1. Begin Story 9.1 by declaring its window-mapping and Child Mind Sequence migration slices under the new delivery rule.
2. Run the live compose test and triage the MLflow signature/deprecation warnings into fixed, accepted, or upstream-owned categories.
3. Execute Warehouse Epics 9–10 with the existing golden and representative baselines as non-regression gates.
4. Record the CAP-7 ADR and generic-spine amendment before Story 10.1 production code.
5. Complete Epic 11 only after the external wheel-only consumer, stable-component approvals, compatibility tests, candidate audit, 50% reduction measure, and 0.3.0 release evidence all pass.

### Final Note

The course correction resolved the requirements-authority, coverage, sequencing, branch-state, and implementation-slicing findings. The remaining work is delivery and release verification: Epics 9–11, the live stack check, and warning ownership. None invalidate the 0.2 spine or the completed tabular/sequence Warehouse cohorts. They do prevent an unqualified claim that the repository has completed its broader Warehouse vision.

**Assessor:** Codex implementation-readiness review
