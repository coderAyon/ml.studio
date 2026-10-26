# Orange/Weka audit repairs — 26 October 2026

This implements repairs requested after `ORANGE_WEKA_DEFECT_AUDIT.md`. That document records the earlier findings; it is not the current feature status.

## Changed behavior

- **Task lists:** Advanced model selectors use the backend task registry (10 classification / 12 regression methods), avoiding a second obsolete list.
- **Preparation handoff:** leaving Prepare data with unsaved settings validates and saves the plan before navigation. Invalid settings keep the user on the current page and report the error. A visible status distinguishes saved/unsaved choices.
- **One preparation engine:** simple training, comparison, workflow and Advanced research share the pipeline builder. Research independently fits the saved plan in each fold. Saved feature exclusions combine with research group/time exclusions. Impact Lab starts from the saved plan and explicitly varies mean filling, scaling or missing flags. Changing the saved plan invalidates the prior research session; previous final-test exposure is still recorded.
- **KNN feasibility:** neighbor count is checked against the actual training split/fold before fitting. Users can change n_neighbors; no silent adjustment is made.
- **Reusable fitted models:** save/open `.mlmodel` on the prediction page or save directly from results. Bundles include the fitted preprocessing pipeline, estimator, metrics and metadata. Version checks, integrity checks, size limits and a fixed audited skops type allowlist are enforced; no pickle loader is exposed. Ordinary project files still do not silently load model code.
- **Prediction import:** prediction controls expose encoding, decimal and delimiter. Direct prediction uses the current dataset import settings when available. Expected input column checking and source/output protection remain.
- **Selectable comparison:** the ordinary training screen accepts 1–6 same-task methods on the same deterministic split and saved preparation. Comparison methods use their own defaults; single-model parameter edits apply to Train.
- **Model settings/search:** optional bounded neighbors, tree depth/estimator count, C, alpha and l1_ratio controls where supported. Small search tests up to three values of one relevant parameter with three folds using training rows only, then refits on the training split and evaluates the untouched holdout. This is bounded search, not comprehensive AutoML. Repeated holdout-driven choices still bias final estimates.
- **Linked scatter selection:** drag a rectangle over up to 5,000 displayed finite points. Selection is calculated over all active rows, not just displayed samples. Inspect original rows, export selected CSV or explicitly create a selected-row dataset version. Undo restores the original active version. PNG/SVG exports retain the separately generated chart; the interactive selection rectangle is not exported into those images.
- **Useful workflow steps:** round Prepare training and Compare models nodes join the existing Dataset, Remove duplicates and Train model nodes. Prepare training validates the saved plan without fitting against all rows; fitting is deferred to training. Compare models runs three starting methods and stores their actual results. This is still a bounded workflow catalog.
- **Portable projects:** newly saved projects embed a typed snapshot of the original imported frame and replay the cleaning log. They reopen without the original CSV and can be saved again. Embedded snapshot integrity is checked. Old projects without snapshots keep their original source-path/fingerprint behavior. Project files contain data and should be shared accordingly.

## Verification

`python -m pytest -q`: **150 passed** (41.86 seconds; 11 expected sklearn fixture warnings). Includes initial red reproductions followed by passing checks for small KNN, training-only search, fold-local saved preparation, model persistence, all registered model roundtrips, positional chart selection, portable project reopening, workflow comparison and non-UTF-8 prediction CSV input with comma decimals and preservation of the input file. Full output: `examples/audit-final-pytest.log`.

Native source acceptance: `python -m mlstudio.audit_fix_smoke`. Evidence in `examples/audit-fixes-verified/result.json` includes saving preparation on Continue, actual custom-parameter training and search, save/reopen prediction equality, exact selected-method comparison, prediction import controls, native mouse scatter selection with subset Apply/Undo, both task lists and the two new workflow nodes.

## Distribution

The project-root `MLStudio.exe` remains the stable source launcher; reopening it loads edits without rebuilding it. The separate `packaging/mlstudio.spec` creates an onedir Windows distribution with Python, Qt/WebEngine, scientific dependencies, UI assets and model persistence. Copy the **whole distribution folder**, including `_internal`, rather than its EXE alone.

A bundled-runtime acceptance run on this computer is not a clean-Windows-VM or second-computer acceptance test. No claim of those external tests should be made without performing them. Deep Learning/GPU, full Orange/Weka catalog parity and unrestricted AutoML remain outside these repairs.

**Bundled acceptance passed:** `dist/MLStudio-Research/MLStudio.exe --audit-fix-smoke <output>` exited 0 with all nine checks, `frozen: true`, working directory set to Windows TEMP and PYTHONPATH/PYTHONHOME removed. Evidence: `examples/portable-audit-final-verified/result.json`. The first build exposed missing skops namespace compatibility imports; explicit hidden imports repaired it, and the rebuilt executable passed. Bundle size is approximately 744 MB. Use `START-HERE.txt` inside that folder.
