# Startup, explicit CSV sources and expanded workflows — 27 October 2026

## Launch behavior

The stable project-root EXE still loads current source; it was not rebuilt. `main.py` now shows an animated launch window before importing the ML engine. Its lightweight Qt helper keeps the orbit/loading sweep moving while the IDE imports libraries. The first rendered workspace triggers a short fade-out. A per-workspace local Qt socket permits one normal application instance: another launch activates the existing window instead of starting a second ML initialization. Smoke/audit command modes remain separate. Startup failures now write a traceback and show its location instead of silently disappearing in `pythonw`.

`packaging/verify_workspace_startup.py` launches the **actual MLStudio.exe twice** under an isolated instance name/storage directory and confirms both launches reach the same application PID. Latest local measurement: workspace rendered in **5.44 seconds**, with 4.70 seconds after the splash helper was started. Timing varies with caches, hardware and system load; this is not clean-machine installer acceptance. Evidence: `examples/startup-verified/result.json`.

`packaging/verify_launch_animation.py` runs the real `pythonw` helper while its owner is deliberately blocked. It verified 61 timer frames, different rendered screenshots, successful ready/fade termination and cleanup when the owner closes its private pipe. Windows' system animation preference is respected. The loading sweep is indeterminate; it does not invent a completion percentage. No artificial minimum loading delay is added. Evidence: `examples/launch-animation-verified/`.

## Explicit source selection

An Import CSV node starts empty even if My dataset already contains a CSV. Its output area always exposes **Choose CSV** and, when relevant, **Use current dataset**. There is no automatic workspace binding. Old nodes without an explicit source choice wait for the user to choose one.

- Choose CSV binds that node to a file, using auto delimiter detection, UTF-8-sig and period decimals. Different file nodes hold independent frames; choosing a file does not replace an existing workspace dataset.
- Use current dataset explicitly links that node to the active workspace version and its reviewed training preparation.
- When no workspace dataset exists, the first node-file import also initializes My dataset so normal project saving and other main tools remain available.
- File-backed nodes use fresh default preparation, not exclusions belonging to an unrelated workspace CSV. File size/mtime changes invalidate cached outputs; unavailable files show an error on their own branch.
- File-backed nodes remain linked to their CSV paths when a project is saved. Keep those files available. The existing portable workspace snapshot does not embed every independent node CSV.

## 38 working steps

Circles are **56 CSS pixels**, down from 72. Port geometry and arrow endpoints match the new size. The library shows its tool count and groups Data, Prepare, Explore, Visualize, Learn and Cluster. Search reveals matching items even inside collapsed categories.

The original 16 steps remain. New steps:

| Area | Added tools |
|---|---|
| Data and cleaning | Filter rows, Sample rows, Drop missing rows, Rename column, Convert to numeric, Trim whitespace |
| Explore | Column statistics, Outlier evidence |
| Training preparation | PCA inputs, Rank/select inputs with F-test, Bin measurements with uniform/quantile bins |
| Clustering | K-Means, DBSCAN, Hierarchical |
| Dedicated learners | Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Linear Regression, Ridge, Random Forest Regressor |

Filtering is literal text containment in a chosen column; sampling uses a chosen fixed record count and seed 42. Learned preparation is deferred to training splits and has an actual before/after numeric preview. Dedicated learner nodes fit their named algorithm and expose the existing real holdout/model outputs. Clustering appends actual cluster labels, supplies internal structure metrics and can feed data tables/charts or CSV export. Because clustering learned from all incoming records, that branch cannot silently feed supervised holdout evaluation; build a separate supervised branch before clustering.

For reference, Orange's [official widget catalog](https://orangedatamining.com/widget-catalog/), [Data Sampler](https://orangedatamining.com/widget-catalog/transform/datasampler/) and [Preprocess](https://orangedatamining.com/widget-catalog/transform/preprocess/) informed the tool categories. ML Studio does not claim their full catalog, plugins, arbitrary joins, complementary sampler output or specialist-domain extensions.

## Verification

- **231 automated tests passed**, including every new tool, independent CSV source isolation, explicit empty source behavior, unchanged input data, branch-local evaluation guards and large unused workspace handling.
- Expanded workflow desktop acceptance passes nine checks: empty import despite a loaded workspace, visible file action, 38 tools, 56px size, independent file import, distinct source row counts, actual statistics, clustering labels and collapsed-category search.
- Existing reactive workflow desktop acceptance passes nine checks for real file import, native port connections, node movement, prepared values/export, training/model activation, graph saving and Visuals routes. Native arrow-suggestion acceptance also passes.

Evidence: `examples/expanded-flow-verified/`, `examples/expanded-flow-regression-verified/`, `examples/flow-suggestions-verified/`. Existing source/runtime limits remain; this does not add GPU/deep learning, general Python/R plugins or complete Orange/Weka compatibility.
