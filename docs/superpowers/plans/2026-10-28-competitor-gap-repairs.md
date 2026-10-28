# Orange/Weka gap repair plan

**Goal:** Check current ML Studio against official competitor catalogs and ship working, offline tabular-data tools for the largest verified gaps.

**Design and authorization:** The user's standing requirement is a clean student interface retaining advanced functionality, with explicit missing-value application and the same root EXE launcher. They requested implementation and asked to continue. Reuse the Python/PySide6 embedded HTML stack. Do not install Orange/Weka or claim full catalog parity. Specialist extensions require separate domain designs.

**Architecture:** Pure engines handle ARFF, association rules, held-out diagnostics and bounded table operations. A small coordinator exposes background actions, invalidates stale results and protects output paths. New pages live under More tools/Results; two-input workflow nodes are real data ports, not menu placeholders.

## Tasks

- [x] Compare official Orange widget catalog and Weka documentation with current source; retain source URLs in a coverage audit.
- [x] Dense ARFF import/export: numeric/nominal/string/missing round trips, useful errors for unsupported sparse/date/relational formats, import/export dialogs.
- [x] Association rules: exact support/confidence/lift, explicit category semantics, bounded Apriori, controls/results/CSV export.
- [x] Diagnostics: recorded held-out ROC/PR/calibration and regression residual plots, fingerprint validation, class selection and exports; no fitting on test rows.
- [x] Merge/concatenate/group/pivot: input-order semantics, explicit keys, missing/duplicate guards, output limits, two-input DAG execution/cache/UI.
- [x] Unit regressions plus embedded UI acceptance covering import → compute → inspect → export and stale-result handling.
- [x] Update accurate coverage and usage docs; commit/push current source to existing main as authorized.

## Review focus

1. Changed dataset/model must not display old diagnostics or rules.
2. Numeric category mining is opt-in; no inferred binning.
3. Merge keys cannot silently multiply rows or match missing keys.
4. Unsafe/unsupported ARFF content is rejected with context and original files preserved.
5. The root launcher loads source; no clean-machine/complete-parity claim without evidence.

## Completed verification

Full suite: 271 passed. Actual project-root EXE: 10 native acceptance checks passed. Independent review found two diagnostics schema identity issues (column name swaps and Boolean/integer type changes); both have regression tests and are fixed. Complete Orange/Weka or specialist-extension parity remains outside the shipped feature inventory and is documented explicitly.
