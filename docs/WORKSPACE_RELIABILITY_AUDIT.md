# Cross-dataset workspace reliability audit — 27 October 2026

This audit tests general tabular workflows rather than treating one Iris example as evidence that the IDE is complete. It does not claim full Orange/Weka feature parity or that all defects are eliminated.

## Reproduced and repaired

| Area | Reproduced problem | Current behavior |
|---|---|---|
| Nullable integer cleaning | Median 1.5 could not fill an `Int64` column | Promotes to nullable `Float64`; preview reports the value and dtype change |
| Category/Boolean cleaning | An entirely empty categorical field rejected the new Missing label; Boolean filling used a numeric median | Adds the category safely; Boolean mode filling preserves Boolean values |
| Entirely missing manual numeric filling | Median filling could silently leave all gaps unchanged | Gives an actionable constant/training-preparation choice; automatic training imputation remains available |
| Wide input tables | More than 201 source columns blocked training even when nearly all inputs were excluded | Checks the 200-input limit after exclusions; 205-source-input fixture trains using two retained inputs |
| Training outcome errors | Infinite classification outcomes and unsupported labels leaked low-level errors | Gives an outcome-specific correction or task/type-conversion message |
| Prediction schema | Duplicate input names and infinite numeric inputs reached transformer errors | Validates unique names and numeric infinities before prediction |
| Workflow preview caching | Changing the outcome/method reused a numeric preview fitted for the old configuration | Invalidates the affected preview; revised output excludes the current outcome |
| Unsupervised preparation branches | Dropping the workspace outcome blocked Normalize's preview | Branches without that outcome can inspect numeric preparation without choosing a supervised target |
| Dataset identity | Reimporting different data at the same file path/version could retain obsolete search/visibility/selection | Uses a session dataset-instance key and clears stale view state on replacement |
| Row selection | Duplicate index labels could make selection ambiguous; selected rows had no usable action | Uses active positions and offers Preview keeping selected rows; Apply is undoable |
| Product information | Settings incorrectly listed fitted-model persistence as unavailable | Lists implemented tools and actual local-engine limits, separately from remaining gaps |

## Verification scope

The full suite passes **205 tests**, including a new 26-case cross-dataset group. The new matrix profiles, analyzes, trains and predicts with two model families across numeric, nullable integer, text-only, categorical, nullable Boolean, timezone-aware date, all-missing numeric input, constant, zero-IQR/outlier and duplicate-index fixtures. It checks source preservation and split row accounting. Targeted cases additionally cover retained-feature limits, error recovery, exact preview types, prediction schemas, workflow cache changes and preview/apply/undo consistency.

Native desktop acceptance replaces the dataset three times, including replacement at the same source path. It checks duplicate-index row selection, selected-row preview/apply/undo, stale state clearing, actual categorical-only model training, nullable cleaning previews and current capability information. Evidence: `examples/workspace-matrix-verified/result.json`. The separate outlier and native arrow-suggestion UI acceptances were rerun with the new identity/selection layer.

These are deterministic synthetic fixtures for correctness. They do not establish accuracy on arbitrary real-world data, novice usability superiority, or performance on every Windows computer.

## Current bounds and open work

- Simple training: 8–20,000 rows and 200 retained inputs. Prediction CSV: 20,000 rows / 300 columns. Clustering: 3–10,000 rows / 100 numeric inputs. Workflow/preparation previews: 20,000 rows / 300 source columns. KNN filling has a separate 2,000-row / 50-numeric-input bound.
- Table pages, chart sampling and numeric previews are bounded and labeled; a preview is not the full exported dataset. Some charts remain static images; scatter selection exists in the supported selection tool, not as universal linked brushing across every chart/node.
- Workflow remains a typed 16-step single-input catalog. Arbitrary fan-in joins, multiple independent CSV sources per graph and universal Orange/Weka extension compatibility are not implemented.
- Missing-value filling, scaling and feature reduction are fitted inside training splits. Whole-dataset exploratory learned changes must be undone before model evaluation. Repeated selection on one holdout is still exploratory; stronger protocols remain in the research tools.
- Deep Learning/GPU, arbitrary Python/R plugins, association-rule mining and a general text/image/time-series feature ecosystem remain outside the implemented scope. Date inputs in ordinary supervised pipelines are treated as categorical labels unless the user engineers suitable features.
- The stable project-root EXE loads current source plus the local environment. A frozen portable build exists separately; clean-machine Windows acceptance/installer validation is still pending.

Review this document and current source/tests instead of older historical inventories when assessing feature availability.
