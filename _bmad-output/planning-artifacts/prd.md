# DSio Product Requirements Authority

## Current product baseline

DSio 0.2 is the completed generic experiment spine defined by:

- `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md`
- `_bmad-output/planning-artifacts/epics.md`

DSio 0.3 / Component Warehouse v1 is the active planned release defined by:

- `_bmad-output/specs/spec-component-warehouse/SPEC.md`
- every companion listed in that SPEC's frontmatter
- `_bmad-output/planning-artifacts/epics-component-warehouse.md`

The Component Warehouse SPEC is the canonical contract for Warehouse v1. Its companions are
normative where the SPEC references them.

## Delivery state

- Generic spine, Epics 1–5: complete.
- Warehouse foundation and tabular/sequence capability work, Epics 6–8: complete.
- Warehouse multimodal, task-spine, and release work, Epics 9–11: backlog.
- Deferred Warehouse work is governed by
  `_bmad-output/specs/spec-component-warehouse/roadmap.md`.

## Precedence

1. The Warehouse SPEC and its companions govern Warehouse v1/0.3.
2. The accepted generic-spine specification governs the underlying 0.2 architecture unless
   explicitly amended by the Warehouse SPEC.
3. `README.md` documents shipped behavior but does not override either canonical contract.
4. Earlier dated design documents are implementation history where marked superseded.

## Non-goals

This index introduces no requirements. It exists only to make the authoritative requirements
discoverable and prevent superseded designs or README examples from being treated as unapproved
scope.
