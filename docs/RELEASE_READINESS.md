# Release readiness — 29 October 2026

ML Studio is a working offline tabular workbench with a separate descriptive Business analysis mode. This checkpoint strengthens correctness without replacing Python, Qt WebEngine, JavaScript, scikit-learn or SQLite. It is **not yet a public commercial release sign-off**. Complete Orange/Weka parity, universal model quality and zero defects are not acceptance criteria we can honestly certify.

## Repairs in this checkpoint

| Risk | Implemented behavior | Verification |
|---|---|---|
| R4: explanations accept a renamed schema | Diagnostics and permutation importance share `holdout.validate_holdout`: ordered columns, dtype/nominal domains, values/index, exact integer holdout positions/count and fitted feature order must agree. Validation does not fit or predict. | `test_holdout_contract.py`, existing diagnostics/student tests; native valid explanation and rejected incompatible evidence. |
| Stale explanation after changing the active data | Only the explanation is invalidated on revision change; model-only prediction and imported prediction tables remain available. Completion is guarded by model and revision identity; a failed explanation clears the previous output. | Real Qt bridge regression and native acceptance. |
| R5: categorical cleaning preview crashes | Compare values per column with explicit missing equality, independent of changing nominal domains. Preview remains read-only; apply, undo and redo retain versions and original source bytes. | Nominal trim/lowercase/all-empty fill; unchanged nullable columns; native preview/apply/undo/redo. |
| R6: quality badge/detail disagreement | Analyzer and row inspector use shared positional text masks. Counts are actual nonnumeric rows or rows in ambiguous normalized spellings, rather than rounded fractions or all present labels. Missing entries and repeated index labels are covered. | Object/string/category matrix, exact badge/detail/position assertions; native 40-row equivalent-label finding. |
| R7: incomplete model manifest passes | Require all four package versions, supported schema, valid feature/target/holdout identity, task metric and seed; compare fitted input order, registered estimator class and actual parameters. Reject duplicate JSON/archive fields and nonfinite JSON constants. Fitted state is checked before acceptance; overflowed JSON numbers are rejected throughout the manifest. Invalid saves preserve the existing destination; failed loads retain the active safe model. | 28 bundle tests; existing roundtrip tests exercise all 22 registered estimators; native model-only prediction and invalid-bundle rejection. |
| R9: research underestimates dense preparation width | One shared bound uses the actual 100-category encoder cap and possible numeric missing-indicator columns. Research reserves space for its indicator-impact variant too; the guided trainer retains its older conservative cardinality guard. | Previously accepted over-budget 80-level categorical and numeric-indicator cases now reject before fitting. |
| R10: research reports constructor defaults instead of actual settings | Comparison/impact rows record the effective fitted estimator parameters and resolved variant preparation. | A real seed-17 experiment records 17 rather than `None`; existing independent research replay passes. |
| R11: legacy executor silently deduplicates unimplemented nodes | Data table preserves all rows. Unsupported legacy operations reject before execution and direct callers to the current reactive canvas. Existing explicit deduplication still works. | Legacy table/unsupported-step regressions and current reactive-node suite. |
| Boolean research input mismatch | Research and guided training now agree that Boolean features follow the stateless categorical preparation path; plain bool input no longer reaches an unsupported Boolean imputer. | Real bool/nullable-Boolean CV with source-frame preservation. |
| SQLite connection lifecycle | History and final-exposure operations explicitly close connections after commit or rollback. | A real SQLite regression verifies persisted records, failed-JSON rollback and absence of unclosed-database warnings under garbage collection. |
| Dependency lock omits model persistence | Windows lock now includes installed `skops`, `prettytable` and `wcwidth` pins. | Current-environment no-index pip dry run and `pip check`; fresh installation remains a separate gate. |

Earlier increments repair R1/R8 revision-cache identity and R2/R3 typed project/research snapshots. These bounded regressions address the eleven reproduced baseline findings; they do not constitute an exhaustive audit of every possible workflow or data type. The baseline audit remains historical evidence.

Safe models remain skops bundles with a fixed application-controlled trusted-type allowlist. Pickle/joblib loading is not introduced. Checksums and structural agreement detect corruption/inconsistency; they are **not publisher signatures or proof that someone else's reported scores are honest**. Older incomplete bundles now require retraining. No model was refitted to manufacture missing holdout evidence.

The dense element budget is a conservative preflight guard, not a measured peak-RAM limit. It includes intermediate width before PCA/F-test/constant removal. It does not promise chunked processing or arbitrary dataset sizes.

## Business mode delivered alongside these repairs

Switch between the existing ML workspace and Business analysis without discarding the active dataset. Implemented computation includes reviewed measure mappings, filters/cross-filtering, KPI totals, paired revenue/cost profit, segments/Pareto, UTC calendar trends, two-dimension pivots, target/arithmetic scenario analysis, source-row drill-down, six HTML/CSV exports and report definition/project roundtrips. Dataset name and its confirmed delete action sit directly below the sidebar brand in both modes. See [Business mode](BUSINESS_MODE.md) for formulas, tested bounds and official research references.

Eight native checks include 48 page/theme/width layouts, real segment click/reset, export/open and source-preserving deletion. This is descriptive BI, not a claim of a full enterprise semantic model or forecasting system.

## Gates still required for a public launch

| Gate | Current status | Acceptance needed |
|---|---|---|
| Clean-machine distribution | Open | Build a self-contained package/installer; install, launch, save/reopen, train and predict on a Windows machine without Python or this checkout, while physically disconnected. Uninstall/update must preserve user projects. The root EXE currently launches local source and `.venv`. |
| Reproducible build and supply chain | Partial | Verify a new lock-only environment, retain offline wheels and notices/licenses, automate Windows build/test checks, choose signing and update/rollback strategy. Existing-environment success is insufficient. |
| Cancellation and crash recovery | Open | Cooperative cancellation across training/research/workflow/exports, cleanup under failure, recoverable autosave, bounded persistent job/node logs. |
| Resource and scale acceptance | Partial | Representative wide/high-cardinality/missing/group/time datasets, measured peak RAM/runtime, long-session retention, allocation-failure recovery and explicit supported hardware budgets. |
| UI/accessibility | Partial | Existing Light/Dark responsive/native tests pass; keyboard-only workflows, screen-reader names, high DPI/multiple displays and sustained responsiveness during expensive work still need acceptance. |
| Study design and tuning | Partial | Training-fold preprocessing and research group/time protocols work; randomized, nested and group/time-aware tuning remain to implement and verify. |
| Project portability | Partial | Original embedded data and reviewed cleaning/graphs/configuration persist; independent workflow file-source portability and external-file freshness need stronger coverage. |
| Product support and release operations | Open | Define supported user workflows/hardware, beta acceptance matrix, privacy/error-report opt-in, support documentation and release/rollback process. |

Do not label any of these gates complete because a button or dialog exists. Prioritize distribution/recovery/performance acceptance before another broad catalog of algorithms. A broad architectural rewrite remains unnecessary and unapproved.

## Evidence

**Final verification:** 453 tests passed, 14 existing warnings, 61.26 seconds; actual root EXE passed 19 checks. Business and ML layout acceptance each cover 48 layouts (96 combined). No unclosed-SQLite warning appears in the final run.

Current checkpoint command results and native check names are saved in `examples/release-readiness-2026-10-29/verification.json` and `native-result.json`. Business evidence is under `examples/business-verified-2026-10-29/`. All fixtures are synthetic; no user CSV was modified and no external pickle/joblib was opened. Historical test totals in older notes should not be added together.

Independent review reproduced two residual model-file gaps (unfitted components and exponent-overflow JSON numbers). Both received failing regression tests before repair. Re-review passed 86 targeted tests, including all 22 estimator roundtrips and advanced preparation, with no remaining blockers in that scoped review. This is code-review evidence, not installer or whole-product certification.
