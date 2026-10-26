# Stronger standard Light theme

Light mode now uses a slate-gray workspace, dark navy text, clearer secondary labels, visible card/input borders, tinted section headers and stronger purple selection/primary actions. Disabled controls remain visibly disabled without fading all their text. Corners and shadows are restrained for a more conventional desktop UI. The override is scoped to Light only; Dark and High contrast light keep their existing palettes. No features or calculations changed. Native screenshots and representative text contrast measurements are in examples/light-standard-verified/. Restart the same project-root EXE to load the styling.

# Dedicated dataset visuals

Dataset visuals appears directly below My dataset and is also linked from the dataset table. It provides direct Scatterplot, Histogram, Boxplot, Bar chart and Correlation tabs without writing a question. Numeric measurements, categories and summaries are editable. Customization includes title/axis labels, palette, opacity, scatter size, histogram bins, grid, bar orientation/order/top groups and expanded chart viewing. Charts and summary tables can be exported as PNG/SVG/CSV.

Outlier evidence supports finite-value 1.5 × IQR and absolute Z-score >3 (population standard deviation), with highlighted scatter points/histogram bounds, rule limits and an exact flagged-row preview. Boxplot points retain their conventional IQR interpretation. Counts use all active rows; at most 50 flagged rows are displayed, and all flagged records can be exported separately. Scatter shows at most 5,000 deterministic sampled finite pairs. Flagging does not remove records or train an anomaly model.

Correlation supports Pearson, Spearman and Kendall, sorted pair coefficients, pairwise finite-row counts and matrix-table export. Kendall is limited to 10,000 active rows; correlation matrices use at most 20 selected numeric columns. Missing/infinite values are excluded per numeric calculation, constant/insufficient coefficients remain blank, and association does not establish causation. General dataset visuals limit: 250,000 active rows. Changing dataset/version invalidates the previous plot; ordinary projects do not persist its session settings or figure. Question-driven visuals and all earlier tools remain available. Native evidence: examples/dataset-visuals-verified/.

# Easier full-data viewing, train/test and round workflows

Dataset table controls now expose Show all columns, searchable column choices, select/hide matching columns, a first-five shortcut, wrapping and an expanded full-preview mode with sideways scrolling. All data columns are included by default; the table still paginates 100 rows at a time and searches all rows. Viewing selections never alter the dataset or training features. Row identity columns remain pinned while scrolling sideways. A document-fragment rendering issue that could leave the prior screen visible when opening Dataset is fixed.

The prediction screen puts the answer column first and suggests a category/number goal from its recorded type/examples/cardinality. This is a reviewable heuristic, not a claim to know a column's meaning; manual goal selection remains available. One button runs both training and held-out testing. A visible record-count plan and an optional 10–40% test share agree with the real training/comparison metadata. Existing advanced validation tools remain available for group/time data. Workflow training still uses its own default holdout protocol.

Workflow nodes are circular with larger ports. Click or keyboard-activate library steps to add them, with automatic connection after a selected data-producing step. Connect two steps also works by clicking entire nodes; cycle and occupied-input checks explain invalid connections. Existing drag/drop, move, port connection, starter flow, layout, zoom, minimap, properties, delete, graph Undo and execution remain accessible. Optional arrangement controls and settings fold away to keep the canvas clear. Native evidence: examples/easy-verified/.

# Student workspace presentation

The desktop now presents the same tools as a student learning workspace: familiar navigation labels, a four-step learning path on the main working pages, brief page introductions and expandable technical explanations. A Quick help drawer contains page-specific guidance and a searchable glossary of ML terms. It supports keyboard focus, Tab navigation and Escape to close. Detailed detection limits, cautions, chart examples, result settings and specialist tools remain available; no computation or feature was removed by this presentation update. Expanded explanations persist through status refreshes during the session. Light, Dark, high contrast and reduced motion remain supported.

Native verification covers control identities across all 18 pages, the actual training/question/outlier flows, the searchable help drawer, preserved disclosures and Light/Dark screenshots. Representative normal text is checked for 4.5:1 contrast; this is scoped evidence rather than a complete accessibility certification. Evidence is in examples/student-verified/. Restart the same project-root EXE to load the UI.

# Minimal workspace and dedicated preprocessing

The sidebar now has seven everyday destinations: Overview, Dataset, Preprocessing, Visualize, Train / compare, Results and Advanced tools. Existing research, workflow and reporting sections remain accessible through the hubs. Dataset inspection is separated from cleaning; detailed row/column information stays available in a disclosure below the table.

Preprocessing lists actual findings on the left. Selecting an outlier, missing-value, duplicate or label finding opens exact matching rows on the right (up to 30 rows / 20 columns), with retained indices and active row positions. Outlier evidence shows finite-data quartiles and IQR limits. Before/after changes stay inline, with counts for affected rows, changed cells, missing entries and removed rows. Reviewed operations create undoable dataset versions and replay through project files. Capping/removing IQR outliers and dropping missing rows are available as explicit operations; detected extremes are not automatically treated as mistakes.

The training plan supports mean/median numeric filling, standard/robust/no scaling, keeping outliers or training-only IQR capping, and explicit input feature exclusions. Category mode/one-hot encoding remains automatic. Learned preparation is fitted only on training rows and reused for prediction. The plan is saved with ordinary projects and used by simple training, its three-method comparison and workflow training; Advanced research uses its own declared fold settings. Whole-dataset imputation/IQR transformations must be undone before predictive evaluation. Training-only zero-IQR features remain unchanged.

Train / compare starts with task and outcome selection, then one primary Train button and a default starting method. Other algorithms and a three-method exploratory comparison are optional. Main metrics appear first; detailed metrics remain expandable. Outcomes are not guessed. Visual requests now also support numeric/ISO-date trends with separate series, category percentage shares with an explicit denominator, and up to six separately reviewed questions.

The contest explanation and short demo plan are in [docs/PRODUCT_VALUE.md](docs/PRODUCT_VALUE.md). No claim of exclusive features or proven superiority over other tools is made. Restart the same project-root EXE to load the latest local source.

# Requirement-driven visual summaries

Visual Analytics now starts with a local visual request builder: write one requirement using the actual CSV column names, click Understand requirement, review/edit the plan, then Calculate visual summary. English plus a supported set of Bengali/Banglish keywords are matched with deterministic rules; this is not an unrestricted language model. Ambiguous or unsupported requests show interpretation questions. A manual plan editor remains available.

Supported results: grouped bar charts with mean/median/sum/count/min/max/sample standard deviation, vertical subplots per group, histogram, scatter, box plot, correlation heatmap and missingness. Controls include a single explicit row filter, sorting/top-N groups, bar direction, bins and custom title/axis labels. Written filter conditions must be specified in the controls; the parser does not execute arbitrary code or interpret arbitrary conditions.

Results include actual summary tables, calculated observations, group sizes and finite-measurement counts, dataset/version and row/missingness/sampling details. Up to six result cards remain in the current session, each exportable as PNG, SVG, CSV, offline HTML report or JSON plan/result. JSON does not include the dataset. The board clears on dataset/version changes; ordinary project files do not persist it. Export reports before closing. Requests support at most 250,000 active rows, 30 displayed groups, 12 grouped measurements, 20 correlation columns and 40 missingness/box columns. Scatter plots show at most 5,000 deterministic sampled finite pairs; correlation still uses all finite pairs. Old Lab studio/chart tools remain under More visual tools. Close and reopen the unchanged project-root EXE to load the update.

# Dataset information update

Dataset Explorer now includes a beginner overview of records, fields, complete rows and rows with missing cells. Expand dataset facts for cell totals, duplicate definitions, empty rows, constant/empty columns and memory. The column explorer includes recorded/distinct counts, examples, contextual review advice, finite-value mean/median/spread/quartiles, zeros/negative/infinite counts and frequent-value percentages. The row inspector accepts a 1-based position in the active dataset and shows all values, types, missing flags and the retained index, independently of table filtering/sorting. Profiles cover all active rows; row details clear when the active dataset version changes. Detailed information stays collapsed initially. Close and reopen the existing EXE to load the interface update.

# Group summaries and lab charts

Visual Analytics now opens with Lab studio. Import a CSV, choose a grouping column and numeric measurements, then use Average by group, Count each group, or Feature means per group. The latter draws one vertically arranged subplot per observed group, covering the three tasks in the supplied lab sheet. Custom chart titles and axis labels, mean/median/sum/min/max/sample standard deviation, summary CSV and PNG/SVG chart export are available. Calculations include all dataset rows (maximum 30 groups and 12 selected measurements); missing group labels are retained and invalid measurements are excluded per feature. Results clear when the active dataset version changes. Uses the same project-root launcher after restart.

# Appearance update — Light, Dark & motion

Light and Dark now cover the complete embedded interface, including tables, workflow canvas, popups, chart labels, research pages and native advanced-import controls. The sun/moon button beside Help switches themes; Settings also offers Light, Dark and High contrast light. Theme preference persists locally. Dark mode is enabled for the next normal application launch.

Light mode uses darker navy text, stronger labels/active states, white cards against a tinted workspace, clear borders and restrained shadows. Page entry, card appearance, hover/press feedback, result plots and popup entry have short transitions. Status updates do not repeatedly animate the page. Reduce motion is available in Settings and the system motion preference is respected. Input drafts, keyboard focus and table scroll survive status/theme updates.

Verification: 56 automated tests passed, the existing 35 core end-to-end checks passed through the unchanged EXE, and native acceptance checked all 15 available pages in Light and Dark, popup/chart rendering, saved preferences, motion and focus behavior. Representative normal text measurements had a minimum contrast of 5.20:1; this is scoped evidence, not a blanket accessibility certification. Screenshots and measurements: examples/appearance-verified/. Existing contest capability limits still apply.

Close and reopen the same project-root MLStudio.exe to load the update. No EXE rebuild is required.

---

# Verified functional status

The contest-focused audit is documented in [docs/CONTEST_AUDIT.md](docs/CONTEST_AUDIT.md). Fifty automated tests and 35 native core end-to-end checks passed, alongside clustering and research acceptance. Fixes cover failure recovery, file preservation, cleaning boundaries, infinity, training limits and reliable interface bootstrap. Deep Learning is explicitly disabled as unavailable. The launcher is still local-development delivery: standalone installation and clean-machine acceptance, trained-model persistence and Deep Learning/GPU are not complete.

---

# Beginner interface & clustering update

Model Lab now offers Classification, Regression and **Clustering — find similar groups** in the task selector. Clustering lets users choose numeric measurements, group count, optional standard scaling and DBSCAN radius/minimum samples. IDs, common outcome names and the currently selected target start unchecked. Supported methods are K-Means, DBSCAN and Hierarchical. Results include a real PCA group plot, group sizes and observed feature means, noise/coverage, internal structure scores, a same-feature method comparison, a K-Means 2–8 group-count exploration and CSV export with group labels. All analysis is exploratory; scores are not prediction accuracy or proof that the groups are useful. K-Means supports up to 10,000 rows; Hierarchical, DBSCAN and comparison up to 3,000. At most 100 selected numeric features. Silhouette uses at most 2,000 deterministic sampled grouped rows; display points are also capped at 2,000. Original data is preserved; changing dataset versions invalidates current clustering results.

The Overview quality card now opens a keyboard-accessible popup with actual issue reasons, severity, evidence, suggestions and a link to the selected Smart Analyzer finding. The duplicated large histogram/scatter/bar section is removed from Overview; all three charts remain in Visual Analytics. The compact dashboard distribution remains.

Beginner presentation uses short task descriptions, numbered clustering steps, collapsed algorithm guidance, advanced validation/feature settings and optional formula/replay explanations. Smart Analyzer places issues before its long cleaning guidance. A tinted workspace, darker borders and text, and soft shadows separate white cards from the background. Critical warnings and disclosure about exported data remain visible.

Verification: 43 automated tests passed (10.92s). Native acceptance covers the quality popup, task switching, clustering comparison/plot/export and stale-result invalidation. Existing research and Dataset Explorer/export/autosave native flows also pass. The existing project-root EXE launcher is unchanged; close and reopen it to load this update. No app distribution was rebuilt.

---

# Current project entry point — pastel UI

Open **MLStudio.exe in this project folder**. It is a stable Windows launcher that runs current `main.py`, `src/` and local `.venv`. After a code/UI edit, close and reopen the same EXE; no PyInstaller rebuild is needed. Keep the project folder and `.venv` together. This development launcher is not a standalone portable distribution. Older distributions under `dist/` are historical snapshots and do not automatically update.

The supplied pastel reference is implemented using local HTML/CSS/SVG with real dataset counters and charts. Dataset Explorer now searches all rows, sorts and paginates, controls column visibility and row selection, profiles all rows, and previews rename, type correction and text row filtering. Each applied operation creates an undoable dataset version. Smart Analyzer provides a transparent completeness/uniqueness health indicator, issue evidence and severity filters, task-specific advice, multi-step cleaning-plan preview, before/after counts and HTML/CSV/PDF health exports. It does not claim that a high health indicator proves valid data or a good predictive model.

Workflow controls include node search, category collapse/expand, DAG auto layout and a clickable minimap. The executable supports the five research tools described below. Settings include pastel/high-contrast themes, optional 60-second autosave after first project save, next-launch storage selection and a configurable research row budget within the hard ceiling. The engine currently supports CPU; selecting GPU is unavailable. Autosave saves the ordinary dataset/workflow project, not the in-memory research session; export a Research Pack to preserve/replay research results.

Verification: 33 automated tests pass; native pastel acceptance covers full-data search, controls, profiles, rename/undo, bulk preview, real HTML/CSV/PDF exports, workflow tools and autosave. Research acceptance covers all five features, final-use lock and exported replay. Acceptance datasets and screenshots under `examples/` are explicitly synthetic, not benchmark performance claims.

---

# ML Studio

A Windows desktop application for local tabular machine learning, with a custom HTML/CSS/JavaScript interface embedded inside the EXE using Qt WebEngine and QWebChannel. No browser window or local HTTP server is required. Development release **0.4.0**; this is an incremental implementation of the master specification, not the completed product.

## Run

The verified environment is Windows x64 and CPython **3.14**. The installed `python` Windows Store alias may not work; use `py` or the explicit virtual environment executable.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/lock-windows-py314.txt
.\.venv\Scripts\python.exe main.py
```

Initial dependency installation requires internet access, or a predownloaded wheelhouse. Application operation requires no network. No datasets are uploaded; no telemetry or cloud services are included. Python 3.12/3.13 are not tested against this lock. Package metadata compatibility was checked by pip on Python 3.14; `pip check` validates the installed dependency graph. Compatibility sources: [Qt for Python](https://pypi.org/project/PySide6/), [scikit-learn](https://pypi.org/project/scikit-learn/), and dependency wheel metadata in the resolved lock.

## Working features

- CSV import with delimiter detection and encoding, header, delimiter and decimal overrides. A 250 MB input limit prevents accidental oversized imports.
- Virtual dataset table, schema summary and background import/analysis.
- Rule-based missing, duplicate, constant, whitespace, category consistency, mixed numeric, cardinality, IQR outlier, skewness and numeric correlation findings. Findings include methods, evidence, recommendations and risks.
- Cleaning preview and explicit Apply, in-memory version undo/redo, cleaned-copy CSV and transformation-log export. Original source export is blocked.
- Histograms, box plots, scatter, correlation, missingness and category charts, labeled with their row limit.
- Eleven supervised algorithms, training-only sklearn preprocessing pipelines and deterministic holdout metrics, per-class reports, and ROC-AUC when available. Model comparison uses identical splits without declaring an unbiased winner. Experiment metadata includes parameters, dataset fingerprint and heuristic duplicate-split/imbalance warnings.
- Numeric K-Means, DBSCAN, hierarchical clustering and PCA; these are explicitly exploratory full-dataset operations.
- Modern dashboard, searchable data preview, column-profile cards, evidence panels, before/after cleaning modal, metric cards, confusion-matrix heatmap, comparison table, history details, reports and settings. Dashboard numbers come from local application state.
- Drag-and-drop workflow library, draggable nodes, click-to-connect ports, node properties, delete/disconnect, graph undo, zoom/fit, validation and execution status. Three real nodes are available: Dataset → Remove duplicates → Train model; choose target/algorithm in Model Lab or the selected node properties.
- SQLite experiment history in `%LOCALAPPDATA%\MLStudio`, session model predictions, JSON projects and HTML reports. Projects verify source SHA-256 on reopen, preserve import options and replay approved operations.

## Usage and correctness

Import CSV, inspect Dataset Explorer, and review Smart Analyzer findings. Select an operation and column, preview proposed rows, then Apply. Undo returns to the prior version; export writes a separate CSV. Exploratory full-data imputation is blocked before model evaluation; undo it and let the training pipeline handle missing values.

In Model Lab select a target and compatible algorithm, then train. Classification uses stratification; insufficient per-class rows produce an error. Regression targets must be numeric. Model comparison is exploratory: repeated selection on the same holdout biases its estimate. Use independent final evaluation data after choosing a model. Clustering ignores nonnumeric columns and reports this in its protocol.

Prediction input must contain exactly the training feature columns; the target should not be included. Models are kept only for the current session. No untrusted pickle/joblib files are loaded.

## Architecture

`src/mlstudio/desktop.py` hosts the embedded presentation, a restricted command bridge, and external-request blocking; `src/mlstudio/web/` contains all local HTML/CSS/JS assets. `app.py` retains the Qt controller, native file dialogs and background workers; `data.py` handles CSV and versioned transformations; `analyzer.py` contains deterministic findings; `training.py` owns estimators and evaluation; `workflow.py` owns restricted graphs; `storage.py` owns SQLite; `report.py` produces escaped offline HTML. No arbitrary code is evaluated from workflow or project files.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

The tests cover core loading errors, findings, immutable original data, undo/redo, graph ports/cycles/order/execution, training metrics, training-only imputation statistics, offline core training, SQLite, escaped reports and offscreen Qt navigation. See DEVELOPMENT_STATUS.md for actual results and limitations.

Runtime smoke test (uses explicitly synthetic fixtures, writes only to the chosen output directory):

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe main.py --smoke-test examples/source-smoke
# After packaging:
.\dist\MLStudio-Guided\MLStudio.exe --smoke-test examples/packaged-smoke
Remove-Item Env:QT_QPA_PLATFORM
```

The check exercises Qt startup, CSV loading, analysis, model training, schema-validated prediction, chart rendering and HTML export, then writes `result.json`. Offscreen checks do not substitute for interactive Windows acceptance.

Manual Windows acceptance checks:

1. Run the app; visit all eleven sections and check keyboard shortcuts.
2. Import your CSV with missing/duplicate rows; verify the preview and issue evidence against the file. Try preview-row search and Column profiles.
3. Preview trimming, cancel, then apply; undo/redo and export a separate copy. Verify the input file is unchanged.
4. Choose a target and train; inspect real metrics and history. Try an invalid target and confirm a clear error.
5. Drag nodes from the library, or use Starter pipeline. Click a node output and then a compatible input; select nodes to inspect properties, delete/disconnect and undo. Validate and run. Try an invalid connection. Use zoom and fit.
6. Save a project, close and reopen it; confirm versions/settings/workflow reconstruct. Modify a copy of the referenced CSV and verify reopening rejects its changed fingerprint.
7. Draw charts, export HTML/logs and predict a CSV with exactly the expected features.
8. Repeat with networking disabled. This clean-machine offline acceptance remains pending.

## Packaging

```powershell
.\.venv\Scripts\python.exe -m PyInstaller packaging/mlstudio.spec --noconfirm
# With Inno Setup installed:
ISCC packaging/installer.iss
```

The spec bundles Qt WebEngine, WebChannel, renderer resources and the local web assets; it excludes unused Qt WebEngine Quick, torch and test packages. The installer uses a user-writable per-user install directory and provides uninstall. User metadata is stored outside the installation. Packaging does not establish clean-machine compatibility: verify startup, CSV import, fitting, charts and reports on a clean Windows VM with networking disabled before distributing. Inno Setup is not currently available on this machine.

The earlier 0.1.0 native-widget build passed its native Qt smoke test. Version 0.2.0 additionally has a native embedded-UI acceptance check; see DEVELOPMENT_STATUS.md for final packaging verification and size. Inno Setup compilation and clean-machine acceptance are pending; the installer script's configuration has not been verified by an actual installation.

Qt uses the ICU API supplied by supported Windows versions. The build spec excludes unrelated `icuuc.dll`/`icudt78.dll` files that PyInstaller otherwise picks up from the development runtime's Poppler directory. The initial packaged smoke test exposed this DLL collision; excluding the conflicting ICU fixed startup and allowed the packaged smoke test to pass. Do not copy unrelated scientific-tool DLLs into the executable directory. Keep the entire `dist/MLStudio-Guided` directory together; this is an onedir build, not a standalone single-file executable.

## Limitations

See DEVELOPMENT_STATUS.md. PyTorch, persistent model bundles, CV, configurable hyperparameters, workflow cancellation and the full node catalog, comprehensive domain/date/leakage detectors, PDF reports, complete report charts, autosave and large-data streaming remain pending. CSV header presence defaults to enabled and can be overridden; automatic header detection and custom missing markers remain pending. Undo stores full in-memory copies, so large datasets and long cleaning histories may use substantial memory. Some preview, plotting, fingerprinting and prediction operations are currently synchronous. Projects reference the source rather than embed it. Projects with incomplete workflow graphs must be completed before reopening. No advanced unavailable node is advertised as implemented. Comparison results appear in a metrics table and are stored in history; sortable comparisons, model selection and persistent model export remain pending.

## Embedded UI acceptance check

```powershell
.\.venv\Scripts\python.exe main.py --ui-smoke examples/modern-acceptance
# Run against the packaged EXE after building:
.\dist\MLStudio-Guided\MLStudio.exe --ui-smoke examples/modern-packaged-acceptance
```

This runs the real embedded page and command bridge with isolated, clearly synthetic acceptance data. It checks WebEngine startup and offline assets, cleaning preview/apply/undo, workflow creation and execution, real training, six-model comparison and chart display; screenshots of all sections and `result.json` are saved in the specified directory. It does not load test data during normal startup or write these test experiments to your personal history. The web renderer uses an off-the-record Qt profile, a restrictive Content Security Policy, an external-request interceptor and an explicit Python command allowlist.

## Guided CSV workflow (0.3.0)

Normal CSV import no longer displays an options dialog: choose a file and it loads with automatic delimiter detection and UTF-8/BOM defaults. Advanced import settings are a separate opt-in action for alternative encodings, headers and decimals. After loading, Overview displays a histogram, scatter plot (when two numeric fields exist) and top-ten category bar chart automatically. The charts use a deterministic sample of at most 2,000 rows; sample size is labeled. Column selectors update the charts. Full chart studio remains available.

Smart Analyzer now includes a dataset-specific preprocessing plan: observed evidence, reasons, risks, and individual Preview buttons for supported cleaning. Training-only imputation, one-hot encoding and scaling are explained and already performed safely inside model pipelines. Choose the prediction target yourself; it is intentionally not guessed from column order. Choose classification for class labels or regression for numeric amounts. Model Lab explains algorithm assumptions, dataset dimensions and tradeoffs. After comparison it identifies observed leaders using weighted F1 for classification or RMSE for regression, handles ties, explains score gaps and states validation limitations. These are deterministic rules and observed metrics, not an AI oracle or proof of why a model performs better. CV/hyperparameter optimization remain future work.

Launch `ML Studio - Guided.lnk` from the repository root or `dist/MLStudio-Guided/MLStudio.exe`. Earlier EXEs are retained so an existing open session is not interrupted.

## Research tools (0.4.0)

Five connected research aids are now available in the sidebar:

1. **Safety Advisor**: choose the outcome and task, random/stratified, group or chronological validation; choose entity/time columns, explicit feature exclusions, 2–5 folds and 1–3 repeats. Findings cover target copies/strong association, potential subject/time columns, future-information names, imbalance, rare classes and duplicates. These are review prompts, not proof of leakage.
2. **Impact Lab**: compare numeric median + scaling, mean + scaling, median without scaling and median with missing indicators on identical development folds. Categorical imputation/encoding remain fixed. Observed paired fold deltas and runtimes are shown; raw data is unchanged.
3. **Reliability**: sequential CPU repeated CV on development rows, sample SD, per-fold scores, baseline scores, paired score gaps and runtime. Classification uses macro F1 (other fold metrics include balanced accuracy); regression uses RMSE. Group splits preserve entities; time splits preserve ordering and timestamp ties and use one forward pass. Roughly 20% is reserved before any research CV; group/time proportions can differ.
4. **Error Detective**: explicitly choose one final candidate after development experiments, acknowledge the final-use lock, then inspect actual held-out predictions, errors, class/missingness/category/numeric slices and feature records. Groups below 20 samples are labeled small. UI tables show up to 200 rows from a 1,000-record diagnostic preview. Subsequent model revision requires new external final data.
5. **Research Pack**: ZIP export of the active CSV, typed frame JSON, exact splits, configuration, package versions, metrics, transformations/provenance and runnable Python source. It contains potentially sensitive data and original paths; export is local. Replaying validates fingerprints and compares recorded metrics (tolerance 1e-9; timings need not match).

Final-test use is locked within a research session. An SQLite exposure audit keyed to the original file's content persists across restarts and dataset versions; new sessions on previously exposed data require explicit acknowledgement and carry an exploratory warning. This is a practical audit, not a guarantee: editing a source file or importing a related extract cannot make reused observations independent. Existing Model Lab holdout experiments remain exploratory and are separate from the research study; do not treat a research final estimate as independent if earlier decisions used its outcomes.

Research limits: 20–20,000 rows, at most 200 columns, an encoded dense-data budget of five million cells and at most 50 one-hot categories per feature. Invalid/rare labels, missing groups, ambiguous numeric timestamps and infinite inputs fail clearly. Jobs run in the background, with fold progress. Cancellation is not yet available.

Recommended demo: Import → Safety Advisor (select target, exclude IDs, choose split) → Reliability → Impact Lab → choose one final candidate → Error Detective → Research Pack. Current study state/estimators are in memory; run summaries are retained in SQLite. Export the Research Pack before closing if you need complete replay. Ordinary project JSON does not yet restore an in-progress research session.

```powershell
.\.venv\Scripts\python.exe main.py --research-smoke examples/research-source-acceptance
.\dist\MLStudio-Research\MLStudio.exe --research-smoke examples/research-packaged-acceptance
```

Use the root shortcut `ML Studio - Research.lnk` for the new EXE. Inno Setup scripts target 0.4.0; clean-machine installer testing remains pending.
