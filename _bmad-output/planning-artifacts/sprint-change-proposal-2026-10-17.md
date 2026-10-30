# Sprint Change Proposal — Requirements Authority and Warehouse Execution Readiness

**Date:** 2026-10-17
**Project:** DSio
**Mode:** Incremental review
**Proposed scope:** Minor — direct documentation and backlog clarification
**Status:** Approved and implemented
**Approved by:** User, 2026-10-17

## 1. Issue Summary

The 2026-10-16 readiness assessment initially treated the accepted generic-spine design, README, and older dated designs as the governing product sources because no conventionally named PRD was discovered in the planning-artifacts directory. The subsequent course-correction review found `_bmad-output/specs/spec-component-warehouse/SPEC.md`, whose frontmatter and opening text explicitly declare it and its companions to be the complete canonical contract for Component Warehouse v1.

This discovery narrows the problem. DSio does not lack a governing Warehouse contract: the accepted generic-spine specification governs the completed 0.2 foundation, and the Component Warehouse SPEC governs the active 0.3/Warehouse-v1 plan. After the proposal was drafted, repository reconciliation also established that `main` has completed Epics 7 and 8. The remaining delivery gap is Epics 9–11; several readiness-report findings also incorrectly treated README-only shipped behavior as missing Warehouse scope.

Supporting evidence:

- The Warehouse SPEC calls itself the canonical, preservation-validated contract.
- The Warehouse epic document lists that SPEC and its companions as inputs.
- `sprint-status.yaml` on `main` records Epics 1–8 as done and Epics 9–11 as backlog.
- The Warehouse baseline still records 70 banned local definitions and 5,794 non-ingestion lines that later epics must reduce.
- Repository verification passed build, 1,071 default tests, formatting, lint, typing, and all import contracts; the live compose-stack artifact test was not executed.

## 2. Impact Analysis

### Epic impact

- **Epics 1–6:** No scope or status change. Completed story IDs remain stable.
- **Epics 7–8:** No requirement or status change. Their tabular and sequence migrations are complete on `main`.
- **Epic 9:** No requirement change. It is the next delivery epic.
- **Epic 10:** Scope remains valid. Story 10.1 needs wording that makes its ADR record the already accepted CAP-7 decision before production implementation, rather than appearing to approve its own architecture during merge.
- **Epic 11:** No scope change. It remains the release gate after Epics 7–10.
- **No new, removed, deferred, or renumbered epic is required.**

### Story impact

- Story 10.1 acceptance language changes as described below.
- Large future stories gain a common slicing rule so multi-consumer work can land through green, evidence-backed vertical checkpoints without creating new requirement IDs.
- The historical Story 6.4/6.5 implementation-order irregularity is retained without renumbering because both stories are complete.

### Artifact conflicts

- A discoverable PRD authority index is missing from `_bmad-output/planning-artifacts/` even though a canonical Warehouse SPEC exists elsewhere.
- The readiness report must be corrected to use the Warehouse SPEC as the authority for Warehouse-v1 coverage.
- The Warehouse epic document needs the Story 10.1 authority clarification and the slicing rule.
- No UX artifact is required because DSio is a Python library.

### Technical and operational impact

- No production code, API, storage format, dependency, infrastructure, or CI change is required.
- `sprint-status.yaml` remains accurate and requires no edit.
- The outstanding live MLflow/Postgres artifact round trip remains a release-verification action.

## 3. Recommended Approach

Use **Direct Adjustment**.

Create a non-normative PRD authority index, correct the readiness report, clarify Story 10.1, and add a large-story slicing rule. Reconcile the work on `main`, then proceed to Story 9.1 under the existing epic sequence.

### Alternatives considered

- **Rollback:** rejected. Completed Epics 1–6 remain valid and strongly verified.
- **MVP reduction/redefinition:** rejected. Warehouse v1 remains achievable and its canonical scope is explicit.
- **New epic or renumbering:** rejected. It would create traceability churn without adding product value.

### Estimate and risk

- Effort: low.
- Technical risk: low; documentation and planning artifacts only.
- Schedule impact: negligible.
- Scope classification: Minor, suitable for direct implementation by the Developer agent after approval.

## 4. Detailed Change Proposals

### 4.1 Create a discoverable requirements-authority index

**Artifact:** `_bmad-output/planning-artifacts/prd.md`
**Current state:** file does not exist.

**New content:**

```markdown
# DSio Product Requirements Authority

## Current product baseline

DSio 0.2 is the completed generic experiment spine defined by:

- `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md`
- `_bmad-output/planning-artifacts/epics.md`

DSio 0.3 / Component Warehouse v1 is the active planned release defined by:

- `_bmad-output/specs/spec-component-warehouse/SPEC.md`
- every companion listed in that SPEC's frontmatter
- `_bmad-output/planning-artifacts/epics-component-warehouse.md`

The Component Warehouse SPEC is the canonical contract for Warehouse v1. Its companions are normative where the SPEC references them.

## Delivery state

- Generic spine, Epics 1–5: complete.
- Warehouse foundation, Epic 6: complete.
- Warehouse foundation and tabular/sequence capability work, Epics 6–8: complete.
- Warehouse multimodal, task-spine, and release work, Epics 9–11: backlog.
- Deferred Warehouse work is governed by `_bmad-output/specs/spec-component-warehouse/roadmap.md`.

## Precedence

1. The Warehouse SPEC and its companions govern Warehouse v1/0.3.
2. The accepted generic-spine specification governs the underlying 0.2 architecture unless explicitly amended by the Warehouse SPEC.
3. `README.md` documents shipped behavior but does not override either canonical contract.
4. Earlier dated design documents are implementation history where marked superseded.

## Non-goals

This index introduces no requirements. It exists only to make the authoritative requirements discoverable and prevent superseded designs or README examples from being treated as unapproved scope.
```

**Rationale:** Makes the canonical sources discoverable by humans and standard planning workflows without duplicating or weakening the preservation-validated SPEC.

### 4.2 Correct the implementation-readiness report

**Artifact:** `_bmad-output/planning-artifacts/implementation-readiness-report-2026-10-16.md`

Replace the incorrect conclusions that no canonical PRD/vision source exists, that README-derived fixed-size/store-verification/masked-evaluation behavior creates Warehouse scope gaps, that strict requirement coverage is 92.5%, and that completed Story 6.4/6.5 requires immediate renumbering.

The corrected report will state:

- the Warehouse SPEC and companions are canonical for Warehouse v1;
- the accepted generic-spine specification governs the completed 0.2 foundation;
- README-only behavior is shipped-product evidence, not additional Warehouse scope;
- core epics cover 38/38 spine FRs;
- Warehouse epics cover all 17 active Warehouse FRs, with FR5 explicitly withdrawn;
- Story 6.4/6.5 ordering is a historical irregularity and completed IDs remain unchanged;
- the full Warehouse remains not ready because Epics 9–11 are unfinished;
- Story 10.1 authority wording, future story size, live-stack verification, and warning ownership are the remaining planning/release concerns.

**Rationale:** Preserve the valid repository evidence and delivery verdict while correcting scope authority and coverage claims.

### 4.3 Clarify Story 10.1 architecture authority

**Artifact:** `_bmad-output/planning-artifacts/epics-component-warehouse.md`

**Additional Requirement — old:**

> Spine amendment: an ADR defines "universal trainer" as a DSio-owned loop or task-dispatching entry point, and shows the CAP-7 functions are neither.

**Additional Requirement — new:**

> Architecture record: the accepted CAP-7 decision distinguishes a prohibited universal trainer—a DSio-owned loop or task-dispatching entry point—from the permitted fit/export spine functions. Before Story 10.1 adds production code, an ADR must record that accepted decision and amend the generic-spine specification.

**Story 10.1 acceptance criterion — old:**

> Given the spine's "universal trainer" non-goal
> When the function is merged
> Then an ADR defines a universal trainer as a DSio-owned loop or task-dispatching entry point, shows this function is neither, and amends the generic-spine spec and `CONTEXT.md` accordingly.

**Story 10.1 acceptance criterion — new:**

> Given the accepted CAP-7 contract and the spine's "universal trainer" non-goal
> When Story 10.1 begins
> Then an ADR records the already accepted distinction and amends the generic-spine specification before production implementation is merged
> And the fit function is verified to own no loop, task dispatch, flow, task, retry, hyperparameter choice, or callback policy.

**Rationale:** Architectural authority precedes implementation; the story codifies and conforms to an accepted decision rather than approving itself.

### 4.4 Add a large-story slicing rule

**Artifact:** `_bmad-output/planning-artifacts/epics-component-warehouse.md`
**Section:** Delivery rules for every story

**New rule:**

```markdown
- **Large-story slicing.** A story spanning multiple consumers or independently
  reviewable component families must declare its implementation slices before coding.
  Each slice must:
  - deliver one coherent component-to-consumer vertical path;
  - leave all repository gates green;
  - record its own parity, evidence-register, and deletion delta;
  - avoid exposing a partially supported stable API.

  The parent story remains `in-progress` until every acceptance criterion and consumer
  migration is complete. Slices are implementation checkpoints, not new requirement IDs.
```

**Rationale:** Makes large migration stories reviewable and reversible without renumbering the backlog or fragmenting requirement traceability.

## 5. Implementation Handoff

### Classification

Minor direct adjustment.

### Recipient and responsibilities

**Developer agent:**

1. Apply the four approved documentation edits exactly.
2. Preserve completed story IDs and all sprint statuses.
3. Validate Markdown links and search for stale contradictory wording.
4. Confirm the resulting diff contains no production-code or status changes.
5. Hand off Story 9.1 as the next implementation target.

### Success criteria

- Standard discovery finds `_bmad-output/planning-artifacts/prd.md`.
- The index points to the SPEC and companions without copying their requirements.
- The readiness report no longer reports false Warehouse coverage gaps.
- Story 10.1 makes the ADR a prerequisite record of accepted CAP-7 architecture.
- Future large stories have an explicit slicing rule.
- `sprint-status.yaml` is unchanged.
- No source code or public API changes occur.

## 6. Checklist Status

- [N/A] 1.1 Triggering implementation story; the readiness audit triggered the correction.
- [x] 1.2 Core problem defined.
- [x] 1.3 Supporting evidence collected.
- [x] 2.1–2.5 Epic impact, future dependencies, viability, and order assessed.
- [x] 3.1 Requirements-authority conflict assessed.
- [x] 3.2 Architecture impact assessed.
- [N/A] 3.3 UX impact.
- [x] 3.4 Secondary artifacts assessed.
- [x] 4.1 Direct adjustment is viable: low effort, low risk.
- [x] 4.2 Rollback rejected.
- [x] 4.3 MVP reduction rejected.
- [x] 4.4 Direct adjustment selected.
- [x] 5.1–5.5 Proposal components and handoff defined.
- [x] 6.1 Checklist reviewed.
- [x] 6.2 Proposal consistency verified.
- [x] 6.3 Complete proposal approved by the user.
- [N/A] 6.4 Sprint-status update; no epic/story/status changes.
- [x] 6.5 Handoff confirmed: direct implementation by the Developer agent; Story 9.1 is next.

## 7. Implementation Record

Implemented on 2026-10-17:

- created `_bmad-output/planning-artifacts/prd.md`;
- corrected the readiness report's authority, coverage, sequencing, and recommendations;
- clarified CAP-7 authority and Story 10.1's ADR prerequisite;
- added the mandatory large-story slicing rule;
- reconciled the assessment against `main`, where Epics 7 and 8 are complete;
- preserved every completed epic/story ID and left `sprint-status.yaml` unchanged.
