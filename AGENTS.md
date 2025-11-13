# Agent Operating Guidelines

- Treat this repository as an audit surface: do **not** delete, rewrite, or otherwise make destructive changes to existing code, telemetry, or artefacts unless the request explicitly demands it.
- When experimenting, prefer additive toggles, feature flags, or new modules so previous behaviours remain reproducible.
- Preserve benchmark history and logging formats; if a change might break consumers, call it out in the PR/commit message first.
- Always confirm with the maintainers before removing files, renaming top-level packages, or rewriting documented workflows.
