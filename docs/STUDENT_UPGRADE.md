# Student workspace and evaluation upgrade — 26 October 2026

## Home

The revised Home groups the active dataset and real counts into one restrained workspace surface. A compact header, clear primary import action, stronger hierarchy, purple accents, context-sensitive next step and quieter recent-experiment rows replace the earlier all-flat layout. It retains the quality popup, source inspection, project actions, four workspace destinations, saved run details and model status. Light, Dark, High contrast, smaller windows and reduced-motion settings are supported. No invented runs, performance scores or demo data populate the real application.

## Gaps addressed after the Orange/Weka comparison

The comparison uses official documentation, not a timed side-by-side usability study:

- [Orange Confusion Matrix](https://orangedatamining.com/widget-catalog/evaluate/confusionmatrix/) exposes records behind correct/incorrect predictions. ML Studio now stores a positional manifest for ordinary holdout training, exposes real errors and source input values, and lets a confusion-matrix cell filter the displayed test records. The preview is bounded to 200 records and 20 inputs, errors first. It explicitly reports that bound; a cell count can exceed the stored rows shown.
- [Orange Predictions](https://orangedatamining.com/widget-catalog/evaluate/predictions/) supports prediction tables and class probabilities. ML Studio now previews prediction CSV input, supports an explicit one-to-one column mapping, permits extra columns only after opt-in, shows generated predictions and probabilities when the estimator supports them, and exports the full result. Single-record prediction requires no CSV. Probabilities are estimates, not a calibrated confidence guarantee. Multiple loaded predictors at once are not implemented.
- [Orange Explain tools](https://orangedatamining.com/widget-catalog/) include model/input explanation. ML Studio now has held-out permutation importance on raw inputs, fitted model unchanged, with up to 500 rows, 100 inputs and 3 repeats. It shows mean score drop and observed SD; correlated inputs and small samples limit interpretation. This is not SHAP, individual causal explanation or proof of importance in the real world.
- [Orange Test and Score](https://orangedatamining.com/widget-catalog/evaluate/testandscore/) and [Weka attribute selection](https://waikato.github.io/weka-wiki/performing_attribute_selection/) provide broad evaluation/preparation tools. ML Studio already has fold-local feature selection, shared preprocessing and repeated research validation. This update adds a simple majority/mean baseline to ordinary holdout results, Balanced accuracy and Macro F1 for classification, and a binary ROC curve when scores are available. Multiclass ROC plots, precision-recall and calibration views remain gaps.

Repeated decisions after inspecting holdout mistakes, importance or ROC still require independent final data. Existing Advanced tools keep their reserved final-test design. No accuracy, ease-of-use or feature-exclusivity superiority over Orange/Weka is claimed.

## Bugs found and repaired during verification

1. **Saved model without dataset:** legacy prediction CSV export called `Path(None)` when no training dataset was open. It now permits model-only prediction and still protects source/output paths.
2. **Project snapshot fidelity:** typed JSON originally turned infinities into missing values and rounded binary floating-point inputs. New versioned snapshots preserve infinities and exact float32/float64 values using checked numeric blocks alongside the typed table. Older snapshots remain readable; already-lost precision in previously saved old files cannot be reconstructed.

## How to use

- Train & compare → train → **Understand predictions · errors, baseline & inputs**. Open the confusion matrix or ROC details, filter mistakes and inspect rows. **Explain inputs** requires the same dataset/version used for that model and runs locally in the background.
- My results → **Open model & predict**. Open a saved model or use the current fitted one, choose the new CSV, review/map its columns, explicitly allow extra columns if needed, then **Generate preview** and **Save predictions CSV**.
- **Try one record** accepts a value for each input; a blank follows the fitted missing-value preparation.

Prediction input is limited to 20,000 rows / 300 CSV columns. Probability output has a two-million-cell budget. UI preview shows up to 50 output rows, the first 12 input columns and up to 20 probability columns; export contains the full result. Original input files are preserved.

Holdout diagnostic inputs are included in saved experiment metadata and model bundles. They can contain data from the training CSV; share those local files accordingly.

## Verification

- Automated suite: **155 passed**, 11 sklearn fixture warnings, 42.41 seconds. Output: `examples/student-upgrade-final-pytest.log`.
- Refreshed contest audit: **46 passed checks**, including real CSV import/options, routes, all 22 supervised algorithms, both task comparisons, workflow actions, six chart types, 11 cleaning operations/Undo/Redo, CSV/log/HTML/PDF exports, exact project roundtrip/autosave, settings validation and background error recovery. Evidence: `examples/student-contest-exact/result.json`. This checks specified flows and representative cases, not every possible setting combination or performance scale.
- New prediction/evaluation native acceptance: `examples/student-upgrade-verified/result.json`.
- Home Light/Dark/High contrast and compact-window navigation: `examples/home-quiet-verified/result.json`.
- Dedicated dataset-chart/correlation/outlier and image/table export acceptance: `examples/dataset-visuals-verified/result.json`.
- Repeated research validation, impact, reserved final evaluation/error inspection and ZIP export: `examples/student-research-verified/result.json`.
- Offline visual-question parsing, supported/unsupported requests, chart settings, preserved drafts and CSV/HTML/JSON/PNG/SVG export: `examples/visual-requests-verified/result.json`.
- Rebuilt portable EXE passes the new student/prediction/evaluation acceptance with `frozen: true`, launched from Windows TEMP without PYTHONPATH/PYTHONHOME: `examples/student-portable-verified/result.json`. The frozen archive also contains the exact typed-v3 snapshot implementation.
- The same frozen executable also passes **all 46 refreshed contest checks** in 26.18 seconds: `examples/student-portable-contest/result.json`, including exact project snapshot roundtrip and all registered supervised methods. Executable process exit code: 0.

The project-root EXE remains the same stable source launcher. The separate portable directory is rebuilt for Python-engine updates. A local frozen-runtime check is different from an external clean-Windows-machine acceptance test, which remains pending.

## Remaining scope

Full Orange/Weka filter/widget/add-on parity, association rules, general text/image/time-series pipelines, ARFF/SQL/multi-table import, unrestricted model search, a large workflow catalog, Deep Learning/GPU, linked multiclass evaluation plots and independent novice-usability benchmarks remain outside this update. These gaps are explicit rather than represented by nonfunctional controls.
