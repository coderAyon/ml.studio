# ML Studio compared with Orange and Weka

**Current reliability update (27 October 2026):** See [WORKSPACE_RELIABILITY_AUDIT.md](WORKSPACE_RELIABILITY_AUDIT.md) for the cross-dataset repairs, current limits and 205-test verification. The inventory below includes historical limitations that have since changed.
**Repair update:** This document preserves the original findings. See [AUDIT_FIXES.md](AUDIT_FIXES.md) for subsequent code/UI repairs and current verification evidence.

Audit date: 26 October 2026. Scope: current local source, targeted Python reproductions and the embedded native desktop UI; Orange/Weka behavior checked against official documentation. Orange and Weka were not installed and benchmarked side by side. This audit changes no application behavior. Severity is based on user impact, not algorithm count.

## Confirmed defects and usability failures

### D1 — High: advanced algorithm lists classify seven regression methods incorrectly

Evidence: `src/mlstudio/web/research.js:6` maintains an obsolete five-name regression list. Main Train uses an updated list, but Advanced comparison and Impact Lab still use this old helper.

Native reproduction: select Decision Tree and a categorical outcome; query `researchAlgorithms()`. Classification choices include Extra Trees Regressor, Gradient Boosting Regressor, AdaBoost Regressor, KNN Regressor, Lasso, Elastic Net and Baseline Regressor. Conversely these seven are unavailable in the Advanced regression list. Backend `ResearchSession._check_candidate` rejects a wrong-task choice, so this causes blocked experiments rather than silently training a regressor as a classifier.

Saved evidence: `examples/comparison-audit/ui-evidence.json`.

Fix: expose the backend task registry to every algorithm selector and remove duplicated frontend classifications. Acceptance: exactly 10 classification and 12 regression choices in both main and Advanced selectors; every visible candidate is accepted for its declared task.

### D2 — High: preparation changes can be bypassed without warning

Evidence: `preprocessing.js` changes `prepUI.draft`; only the explicit Save training plan action updates the backend. Continue to Train simply navigates. Native reproduction changed scaling to `minmax`, then navigated to training without saving: draft was `minmax`, backend remained default Standard scaling, navigation was not blocked. Same behavior applies to numeric filling, exclusions and other draft choices.

This is a workflow defect, not incorrect fitting of a saved plan. A student can reasonably believe a selected treatment is active when it is not.

Fix: show unsaved state and make Continue save a validated plan, or require an explicit save/discard choice before leaving. Acceptance: training always displays and uses the acknowledged plan; no silent mismatch.

### D3 — Medium: valid small datasets fail with a raw KNN error

Evidence: `training.py` permits eight rows and test shares up to 40%, but constructs both KNN models with sklearn default `n_neighbors=5` without checking split size.

Reproduction: eight rows, inputs x=0..7, labels alternating 0/1, algorithm KNN or KNN Regressor, `test_size=.4`. Training has four rows. Both fail during prediction with `Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 4, n_samples = 4`.

Fix: validate the actual training/fold size and provide an editable valid k or a clear preflight message. Do not silently change k across folds without reporting it. Acceptance: every accepted setup completes or gives an actionable message before fitting; include small Advanced folds too.

### D4 — Medium: prediction CSV import does not expose the training import controls

Evidence: `app.py:633` calls `load_csv(path)` with its defaults, unlike configurable dataset import. Delimiter detection exists, but encoding defaults to UTF-8-sig and decimal to a period. A model trained from a non-UTF-8 CSV or comma-decimal measurements has no equivalent prediction import setup; a compatible schema can still fail decoding/numeric conversion.

Status: confirmed missing configuration path by source inspection; a full native prediction-file reproduction was not performed in this audit. Fix: preview prediction types, permit explicit import settings and column mapping, and preserve safe input/output separation.

## Product gaps relative to documented competitor workflows

These are missing or limited capabilities, not all software bugs.

| Priority | Current limitation | Documented reference / practical effect |
|---|---|---|
| High | Fitted session model cannot be saved and loaded | Orange Save/Load Model and Weka model saving/loading support reuse. ML Studio project reopening explicitly sets `result=None`; saved metric history is not a reusable predictor. |
| High | Root EXE depends on source and `.venv` | `packaging/workspace_launcher.py:11–14` requires local Python, main.py and src. Copying only the EXE to a judge's machine will not work. Standalone package acceptance is separate and unfinished. |
| High | Simple and Advanced preprocessing configurations differ | `research.py:35–46` supports four separately declared variants; saved KNN/Min–Max/PCA/F-test/exclusions do not automatically transfer. Orange supports routing an explicit preprocessor to evaluation. ML Studio does disclose the difference, but users must manage two plans; differences in scores are not solely algorithm/protocol effects. |
| Medium | Default comparison always chooses three fixed methods | `desktop.py:288` ignores a general candidate set. Advanced offers multi-candidate repeated validation, so cross-validation is **present**, but D1 currently damages its selector. Orange Test and Score accepts multiple learners with shared sampling. |
| Medium | No general hyperparameter editor/search | Both training engines instantiate registry classes with defaults plus seed. Weka documents CVParameterSelection and optional/package-dependent search tools; Orange includes Parameter Fitter. Repeating default algorithms does not establish each algorithm's best achievable quality. |
| Medium | Dataset charts do not support linked brushing | `dataset_visuals.js:24` renders a chart image. Custom axes, styles, outlier tables, PNG/SVG exports are present; selecting a point region and passing those rows into another tool is absent. Orange Scatter Plot explicitly outputs Selected Data. |
| Medium | Workflow is a small fixed catalog | UI exposes Dataset, Remove duplicates and Train model. Graph also recognizes a legacy Fill missing node but rejects it before evaluation. No chart, split/evaluation, feature-selection, prediction or model-export nodes. Node appearance does not imply Orange's workflow coverage. |
| Medium | Project is tied to an absolute source path | `save_project` stores source path/fingerprint and transformation log rather than a portable dataset bundle. Move/delete the source and reopening fails. A strict changed-source check is useful protection, but no relink/recovery workflow is provided here. |
| Lower | Specialist mining and learning tools are missing | Orange catalog includes association mining, explainability and text/image/time-series extensions. Weka has a wider classifier/filter/package ecosystem. These are strategic scope gaps; adding every add-on is not necessary for a coherent student product. |

## What the audit does not call a defect

- Missing-value filling and normalization **exist**: five numeric filling choices, categorical filling, Standard/Robust/Min–Max/MaxAbs/L1/L2 and no scaling.
- Saved simple-training transformations fit only training rows; no test-statistic leakage was demonstrated by this audit.
- Min–Max outputs outside 0–1 for future values outside the training range are expected behavior, not a normalization failure.
- Advanced research already includes group/time protocols, repeated folds, baseline comparison, a reserved final test, error analysis and replayable exports. Calling validation entirely absent would be inaccurate.
- Existing 141-test evidence does not establish absence of defects. D1 and D2 are integration/interaction paths not covered by the earlier new-preparation acceptance.
- Neither offline operation nor train-only preparation is established as unique to ML Studio. Orange explicitly documents fold-local preprocessing.

## Recommended repair order

1. Correct all task-aware algorithm selectors and add integration coverage.
2. Prevent unsaved preparation from being silently bypassed; unify or explicitly transfer declared preparation into Advanced evaluation.
3. Validate KNN/fold feasibility and prediction CSV import settings.
4. Implement fitted-model persistence and verify reopening/prediction, then test a distributable package on a clean Windows machine.
5. Improve selectable model comparison and optional parameter tuning; retain the simple default entry point.
6. Add plot-to-table selection and useful workflow nodes only after the core flow is reliable.

For a contest, reliable import → understand → prepare → evaluate → reopen/predict is more defensible than claiming every Orange/Weka feature. A claim of easier use requires novice task testing (completion time, errors and assistance), which this audit did not perform.

## Official comparison sources

- [Orange Preprocess](https://orangedatamining.com/widget-catalog/transform/preprocess/) — preprocessing methods, configurable order and fold-local application.
- [Orange Test and Score](https://orangedatamining.com/widget-catalog/evaluate/testandscore/) — multiple learners, cross-validation, repeat sampling and separate test data.
- [Orange Save Model](https://orangedatamining.com/widget-catalog/model/savemodel/) — trained model persistence.
- [Orange Scatter Plot](https://orangedatamining.com/widget-catalog/visualize/scatterplot/) — row selection, linked data outputs and interactive exploration.
- [Orange widget catalog](https://orangedatamining.com/widget-catalog/) — core and specialist/add-on coverage; catalog entries do not mean every add-on is installed by default.
- [Weka saving/loading models](https://waikato.github.io/weka-wiki/saving_and_loading_models/) — Explorer and command-line persistence/re-evaluation.
- [Weka parameter optimization](https://waikato.github.io/weka-wiki/optimizing_parameters/) — built-in and optional/version-dependent tuning approaches.
