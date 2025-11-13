# Historical Journals

This directory centralises work journals, investigation logs, and dated benchmark notes so the top-level documentation can focus on the current source of truth. Use it to:

- archive run-by-run notes that used to live inline inside design docs;
- capture regressions or incident timelines with exact commands, telemetry paths, and outcomes;
- reference older experiments without forcing readers to guess whether a section is current.

## Conventions

- Name files by `YYYY-MM-description.md` (e.g. `2025-11-12_residual_dense_regression.md`) when the entry covers a specific day, or `YYYY-MM.md` when summarising an entire month.
- Keep each entry self-contained: include the command, datasets, artefact paths, and conclusions.
- Link back to the journal entry (instead of duplicating its content) when you need historical context inside design docs.

## Index

- [2025-11-12 Residual Dense Baseline Regression](2025-11-12_residual_dense_regression.md)
- [2025-11 summary](2025-11.md)
- [Audit notes](AUDIT.md)
- [Residual audit implementation plan](AUDIT_IMPLEMENTATION_PLAN.md)
