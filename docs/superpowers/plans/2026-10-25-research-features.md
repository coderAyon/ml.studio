# Research Features Implementation Plan

> For agentic workers: execute inline with superpowers:executing-plans; no delegation required.

Goal: add five working, tested research aids to the local desktop application.
Architecture: pure safety/validation/research/error/export modules, a Qt bridge/controller and separate research.js presentation. Existing training remains supported.
Tech Stack: installed pandas/numpy/scikit-learn, Qt WebEngine, SQLite, zipfile and pytest; no new dependencies.
Spec: docs/superpowers/specs/2026-10-25-research-features-design.md

## Global constraints
- Offline, no data upload, no silent transformations, no learned preprocessing outside training folds.
- Positional source-row IDs; deterministic seed 42; reserve 20% final test before development CV.
- No unsupported statistical-significance or causal performance claims.

## Review focus
- Same subject must not cross group splits; timestamps must not reverse across time boundaries.
- Target/group/time/excluded features must not accidentally enter a fitted pipeline.
- Dataset/target/protocol edits must invalidate stale results and final-test claims.
- Rare classes, constant targets, missing groups, malformed times and no features fail clearly.
- Exported code and exact splits must reproduce metrics with dataset fingerprint validation.

## Tasks
- [x] 1. Write safety, grouped/time split, training-only fit and isolation tests; verify expected failures. Implement safety.py and research.py pure APIs.
- [x] 2. Test repeated comparisons, preprocessing paired deltas and final-once evaluation. Implement same-fold sequential CPU experiments with explicit baseline/uncertainty metadata.
- [x] 3. Test real error rows and subgroup counts for classification/regression. Implement error_analysis.py with bounded previews and no unsupported fairness claims.
- [x] 4. Test exported ZIP fingerprint and run replay in subprocess. Implement research_export.py; bundle source modules for frozen distribution.
- [x] 5. Extend bridge with background research commands, state invalidation, local SQLite history and native ZIP export; add research.js pages and working controls. Exercise actual page/bridge through native UI acceptance.
- [x] 6. Run full pytest and JS syntax checks, inspect screenshots, verify the project-local stable EXE launcher, run its full native research acceptance and replay check (updated per user request to avoid rebuilding app distributions). Update README/status and shortcuts only after verification.

User follow-up changed release delivery: use one project-local launcher that executes current source. Full legacy frozen distribution is retained but is not the current delivery artifact. Independent reviewer hit its account usage limit; self-review and documented tests/native acceptance completed.
