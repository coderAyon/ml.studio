# ML Studio feature coverage — 26 October 2026

**Update:** The subsequent [audit repairs](AUDIT_FIXES.md) add model save/load, shared Advanced preprocessing, selectable comparison, bounded parameter search, scatter row selection and portable project snapshots. The original inventory below is a historical snapshot; use the repair document for these changed limitations.

This is an implementation inventory, not a claim of complete Orange or Weka compatibility. Their widget/filter catalogs include many specialist tools and extensions. No usability superiority or benchmark superiority has been established.

Official references consulted:
- [Orange Preprocess](https://orangedatamining.com/widget-catalog/transform/preprocess/)
- [Orange widget catalog](https://orangedatamining.com/widget-catalog/)
- [Weka attribute filters](https://weka.sourceforge.io/doc.stable/weka/filters/unsupervised/attribute/package-summary.html)
- [Weka Normalize](https://weka.sourceforge.io/doc.stable/weka/filters/unsupervised/attribute/Normalize.html)

## Available training preparation

Open **Prepare data → Change training preparation or exclude feature columns**. Extra options are folded under **More preparation**. Click **Save training plan**, then train from **Train & compare**.

| Task | Implemented choices |
|---|---|
| Numeric missing values | Median, mean, most frequent, chosen constant, KNN |
| Category missing values | Most frequent or chosen label; automatic one-hot encoding |
| Numeric scales | Standard, robust, Min–Max, max absolute, L1/L2 numeric-row normalization, unchanged |
| Missing flags | Flags for numeric columns missing during fitting |
| Outliers | Keep or training-only IQR capping |
| Discretization | Uniform/quantile bins, 2–20 groups |
| Feature reduction | Constant input removal, supervised F-test selection, PCA |
| Input control | Explicit exclusions; outcome cannot be excluded or imputed |
| Evidence | Actual prepared input names/count, learned fixed fill values and pipeline steps in model results |

Learned operations fit only on training rows and are reused on held-out/prediction rows. They do not rewrite the CSV. The plan applies to simple training, its three-method comparison and workflow training. Advanced research experiments retain their separately declared fold preparation; this page does not silently override it.

Min–Max uses training extrema; future values may exceed 0–1. KNN filling uses original numeric measurements before scaling (up to 2,000 rows / 50 numeric inputs); differently scaled measurements can distort its neighbor distances. L1/L2 normalize the numeric block per row, not encoded categories. Missing indicators also pass through selected numeric binning/scaling. PCA uses all retained encoded inputs and changes feature axes. F-test is association, not causal importance. All-constant removal and invalid PCA dimensions return errors rather than claiming successful fitting.

## Models

10 classification methods: Logistic Regression, Random Forest, Decision Tree, KNN, SVM, Naive Bayes, Extra Trees, Gradient Boosting, AdaBoost, baseline classifier.

12 regression methods: Linear Regression, Ridge, Random Forest Regressor, Decision Tree Regressor, SVR, Extra Trees Regressor, Gradient Boosting Regressor, AdaBoost Regressor, KNN Regressor, Lasso, Elastic Net, baseline regressor.

Baseline models provide a simple reference; no method is automatically claimed best. Simple comparison still runs three starting methods on a shared holdout, not all 22 methods. Advanced validation remains available for stronger evaluation.

Existing tools also include numeric clustering, editable dataset inspection, cleaning preview/apply/undo, histogram/scatter/box/bar/correlation views, outlier evidence, guided training, repeated validation, error inspection and research exports. These are bounded implementations, not replacements for every Orange/Weka widget.

## Remaining gaps

- Complete Orange widget/add-on coverage (including text/image/domain workflows), interactive linked plot brushing and Weka filter/classifier catalog parity.
- Association-rule mining and a general hyperparameter search interface.
- Applying every new preparation choice to the separate advanced research variant engine.
- Persisting fitted session model bundles, Deep Learning/GPU, standalone clean-machine installer acceptance.

## Verification

`python -m pytest -q`: **141 passed**. Includes every registered supervised algorithm, all numeric fill and scale choices, category constants, flags, discretization, feature selection/PCA, prediction schema, training-only fitted statistics, project reopening and existing regression coverage. Synthetic fixtures are acceptance data, not evidence of real-world model quality.

Native desktop acceptance: `python -m mlstudio.preparation_smoke` verifies control availability, saved KNN/Min–Max/missing-flag/F-test settings, task-specific algorithm lists, an actual trained model using those settings, displayed fitted input evidence and unchanged active data. Evidence: `examples/preparation-verified/result.json` and screenshots.

Root `MLStudio.exe` remains the same stable source launcher. Reopen it to load source changes; the local environment is still required.
