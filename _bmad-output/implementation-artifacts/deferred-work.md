# Deferred work

## Deferred from: code review of 9-4-compose-multimodal-fusion-with-modality-presence-gating (2026-10-18)

- Revisit whether fitted standardization should support a neutral `(mean=0, scale=1)` fallback for a feature with no observations. The current fail-closed behavior was established and tested in Story 9.2.
- Consider recording a separately materialized sequence dataset configuration containing fitted sample weights. The current Story 9.3 design records the dataset config and fitted-weight evidence separately.
