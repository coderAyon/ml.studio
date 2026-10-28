# Orange/Weka comparison and implemented gaps — 28 October 2026

ML Studio does **not** contain every Orange/Weka option. This audit compares official documentation with the local implementation, not installed competitor applications or performance benchmarks. A similarly named scikit-learn method is not necessarily the same algorithm or settings as a Weka implementation (for example, CART versus J48/C4.5).

## Added in this change

| Capability | Where to use it | Actual implementation and boundaries |
|---|---|---|
| Weka data exchange | Import dataset; Reports & exports → Export ARFF | Dense numeric/nominal/string ARFF with missing values, quoting and nominal domains. Workflow and prediction import use the same loader. Sparse rows, DATE/RELATIONAL attributes and instance weights return explicit unsupported-format errors. Date exports are ISO strings. Existing files survive failed export. |
| Association rules | More tools → Association rules | Bounded Apriori, exact support/confidence/lift, optional CSV with additional leverage/conviction. Each row is a transaction of selected column=value items. Numeric categories require opt-in; missing cells add no item and all rows remain in the denominator. Up to 20,000 rows, 20 selected columns, 50 values per column, item/candidate/memory limits. It is descriptive mining, not a validated classifier. |
| Classification diagnostics | My results or More tools → Model diagnostics | Recorded holdout confusion/per-class results; selected-class one-vs-rest ROC/PR, calibration and Brier score when true probabilities exist. SVM decision scores support ROC/PR; they are not converted to fake probabilities. Full evidence CSV and SVG exports. No refitting or threshold optimization against the holdout. |
| Regression diagnostics | Same diagnostics page | Predicted-versus-actual and residual plots, MAE/RMSE/R² and full recorded test predictions export. Large plots sample rendering only; scores/CSV use every recorded test row. |
| Merge data | Workflow → Data → Merge data | Two ordered input connections; explicit left/right key and left/inner/outer join. Missing keys and repeated Input 2 keys are rejected to prevent unexpected row multiplication. Single-key matching; no composite-key or instance-ID mode. |
| Concatenate | Workflow → Data → Concatenate | Two table inputs; union or intersection of column names. Conflicting numeric/category types are rejected. Both originals stay unchanged. |
| Group by / Pivot table | Workflow → Data | Count, mean, median, sum, min, max, sample standard deviation. Missing groups retained. Up to 200 row groups and 100 pivot headings; output bounded by 20,000 rows/300 columns. Aggregated full-data branches are blocked from supervised evaluation. |
| Training preflight | Train & compare, when the chosen outcome has gaps | Inline missing-answer count and a direct link to its Prepare data evidence. Outcome labels remain subject to explicit source correction/review. |

The reactive library now has **42 working node types**. Merge/Concatenate accept two connections; other data inputs keep their one-input contract. Changing either source invalidates the combined output/cache. Missing values still requires an explicit **Impute** click.

## Existing coverage and remaining differences

| Area | Current ML Studio coverage | Still absent or narrower than competitor tools |
|---|---|---|
| Import and inspection | CSV options, dense ARFF, column profiles, row/column previews, search/sort, visibility and width controls, portable project snapshots | SQL connectors, Excel/native Orange TAB metadata, sparse/relational ARFF, streaming and database workflows |
| Cleaning/preparation | Reversible cleaning, five numeric fill methods, category filling, one-hot encoding, missing flags, Standard/Robust/MinMax/MaxAbs/L1/L2, IQR capping, discretization, constant removal | Complete Weka filter catalog, Orange formula/domain/value editing and arbitrary processor order |
| Prediction models | 10 classifiers and 12 regressors, baselines, bounded parameter edits and training-only search, reusable fitted model save/load | Weka J48/rule/Bayes-network/lazy/meta catalogs, Orange CN2, PLS, neural networks, SGD, stacking and calibrated learners; GPU/deep learning |
| Evaluation | Shared holdout comparison, repeated/group/time validation, independent final-test workflow, baseline/error evidence, new held-out curves | Full Orange multi-learner evaluation signal graph, pooled cross-validation curves, cost-sensitive threshold tools and complete Weka Experimenter |
| Clustering/projection | K-Means, DBSCAN, hierarchical clustering, PCA, real cluster metrics and label output | Louvain, SOM, t-SNE/MDS/Isomap, Weka EM/Cobweb and distance-file workflows |
| Visuals | Histogram, scatter with rectangle row selection, box/bar/correlation, grouped summaries, question plans and exports | Violin/mosaic/sieve, general heatmap/line tools, FreeViz/Radviz, tree/forest viewers and linked selection in every plot |
| Feature explanation | Training-only F-test selection/PCA and held-out permutation importance | ReliefF/RFE/complete feature-scoring catalog, SHAP/ICE/counterfactual tools |
| Association | New bounded Apriori | FP-growth, predictive/class association-rule catalogs and complete Weka association settings |
| Specialist domains | General tabular data | Text/image/network/geo/survival/bioinformatics/single-cell/spectroscopy/time-series add-ons |
| Delivery | Same project-root source launcher; existing separate frozen distribution is historical | Up-to-date standalone installer and clean Windows machine acceptance |

## How to run the new tools

1. Import a CSV or supported dense `.arff` file normally. Advanced CSV-specific delimiter controls do not apply to ARFF headers.
2. Open More tools for association mining or model diagnostics. Train/open a model and restore the exact training dataset/version before building its diagnostics. The recorded schema includes names, types and nominal domains. Older models without this new schema record need retraining for diagnostics; their ordinary prediction tools still work.
3. In Workflow, drag two Import CSV nodes, choose their independent files, add Merge data or Concatenate, and connect both outputs. Input order is explained in the right settings panel.
4. Use Group by/Pivot table on a data branch to create actual summary tables. Export a ready output from the inspector.

## Validation

- Full Python suite: **271 passed**, with 12 existing scikit-learn warnings in numerical/preparation fixtures.
- New/related engines cover table operations, ARFF, association, diagnostics and existing reactive/expanded workflow behavior, including rejection of oversized category labels and nested cells.
- Native embedded acceptance: the actual project-root `MLStudio.exe --comparison-smoke examples/comparison-tools-verified` passed **10 checks**, covering real ARFF import, association controls/CSV export, classification curves and class selection, regression diagnostics in dark mode, SVG/CSV/ARFF export, two-input merge/concatenation, group/pivot results, graph project save, stale-result invalidation and missing-answer preflight.
- Evidence: `examples/comparison-tools-verified/result.json` and screenshots. Synthetic acceptance data does not establish model superiority or usability superiority.

## Official sources checked

- [Orange widget catalog](https://orangedatamining.com/widget-catalog/): core and specialist extension categories.
- [Orange Merge Data](https://orangedatamining.com/widget-catalog/transform/mergedata/) and [Concatenate](https://orangedatamining.com/widget-catalog/transform/concatenate/): multi-table semantics.
- [Orange ROC Analysis](https://orangedatamining.com/widget-catalog/evaluate/rocanalysis/) and [Test and Score](https://orangedatamining.com/widget-catalog/evaluate/testandscore/): evaluation signal and curve capabilities.
- [Weka ARFF format](https://waikato.github.io/weka-wiki/formats_and_processing/arff_developer/): numeric, nominal, string, date, sparse and weighted formats.
- [Weka Apriori](https://weka.sourceforge.io/doc.stable/weka/associations/Apriori.html) and [association package](https://weka.sourceforge.io/doc.stable/weka/associations/package-summary.html): mining algorithms and options.
- [Weka Explorer guide](https://weka.sourceforge.io/manuals/ExplorerGuide.pdf): classify, cluster, associate, feature selection and visualization.
