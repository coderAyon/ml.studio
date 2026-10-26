# Dataset visuals below My dataset — 2026-10-26

Added a dedicated Dataset visuals destination directly below My dataset, plus a table-page shortcut. Direct scatter/histogram/box/bar/correlation tabs use real active data. Column selections, graph title/axis labels, palette, opacity, point size, histogram bins, grid and bar display settings are customizable. Chart expansion and PNG/SVG/table CSV exports are available.

Outlier evidence supports finite-value 1.5 × IQR and absolute Z-score >3 (population std); scatter highlighting, histogram limits and conventional IQR box fliers are distinct and explained. Flagged counts cover all active rows, previews show up to50, and full flagged records export separately. Correlation supports Pearson/Spearman/Kendall, sorted pair coefficients and pairwise finite counts. No data is removed by charting or flagging. Figures invalidate when data versions change. Kendall limit10k rows; general visuals250k; correlation at most20 columns; scatter at most5k deterministic sampled finite pairs.

Verification: 102 pytest tests passed (16.88s), JS syntax and native five-tab/correlation/customization/export checks passed. Native evidence in examples/dataset-visuals-verified includes the exact flagged row121, actual coefficients/pair counts, Light/Dark, expand/Escape and exported files. Existing EXE hash unchanged.

# Column preview, guided train/test & round flows — 2026-10-26

Dataset preview now has visible column counts, Show all, searchable selection, matching select/hide, first-five shortcut, wrapping, pinned row identity, horizontal scrolling and expanded preview. It continues to paginate/search all records and does not alter modeling inputs. A pre-existing DocumentFragment outerHTML rendering failure on Dataset was fixed with safe node replacement.

The simple prediction path suggests a reviewable category/number goal after answer-column selection and runs train plus held-out test together. Optional test shares (10–40%) are validated, used by actual training/three-method comparison and recorded in model metadata. Workflow training retains its existing default holdout. Round workflow nodes have larger aligned ports, click/keyboard addition, automatic data-edge connection, whole-node connection with cycle/input checks, and retained drag/drop/movement, graph undo, settings, layout, zoom and minimap. Optional canvas controls are folded rather than removed.

Verification: 92 pytest tests passed (36.28s), JavaScript syntax checks and pip check passed. Native acceptance in examples/easy-verified verifies 14-column viewing/selection/full preview, actual 80/20 train/test, connected round-node execution, native mouse-drag movement and existing drag/drop creation. The existing project-root EXE hash remains unchanged; close/reopen it to load the current source.

# Student-friendly presentation — 2026-10-26

Added student.css/student.js as the final presentation layer. Student-oriented page/navigation labels, a compact learning-step guide, calmer card/typography treatment and progressive disclosure reduce the information presented at once. A contextual Quick help drawer contains page guidance and a searchable ML glossary, with keyboard focus/Tab/Escape support. Full outlier rules, cautions, chart examples and all prior feature controls remain reachable; no calculation was removed or substituted.

Native acceptance in examples/student-verified checks feature-control identities across 18 pages, actual training/question/outlier paths, glossary search/focus, disclosure persistence and Light/Dark screenshots. Representative normal text contrast is >=4.5:1; this is scoped evidence, not full accessibility certification. Source CSV preservation and the project-root launcher remain unchanged.

# Dark theme & interface polish — 2026-10-26

Added shared light/dark color tokens and complete component coverage in polish.css; stronger light text/card separation; persisted dark preference with header toggle; contrast-light compatibility; matched native import dialog palette; Matplotlib chart paper/axis recoloring without altering data; brief page/card/popup/press feedback and reduced-motion support. Appearance-only changes are allowed during training; resource/data mutations remain locked. Status refreshes preserve drafts/focus/scroll and do not replay page-entry effects. Initial document receives saved theme before rendering. Dark is saved as the user's next-launch preference.

Verification: 56 pytest tests passed (15.95s); 35 core native checks passed through unchanged root launcher (examples/polish-core-verified/result.json); all 15 available pages in both themes plus chart/popup/motion/preference checks passed (examples/appearance-verified/result.json). Representative text contrast minimum 5.2004:1. CSS/motion review completed; native screenshots inspected. Root EXE hash unchanged. Existing development-launcher/contest limitations remain documented in docs/CONTEST_AUDIT.md.

---

# Contest functional audit — 2026-10-26

See docs/CONTEST_AUDIT.md for feature-by-feature evidence and remaining contest blockers. Fifty automated tests and 35 core native end-to-end checks passed. The core audit also passed via unchanged project-root MLStudio.exe. Clustering and the five research flows passed their native acceptance suites.

Fixed hidden-window error dialogs; empty workflow validation; empty cleaning versions and atomic bulk validation; infinity handling; protected export destinations; unique save temporary files; prediction-column collisions; normalized rename collisions; UI bootstrap ordering; clear training task/shape/dense-memory limits; misleading workspace arrow and enabled Deep Learning entry. Numeric chart selectors now use numeric columns. Deep Learning is explicitly unavailable.

This is not yet a complete contest deployment: Deep Learning/GPU, saved trained-model bundles, installer and clean-machine verification remain pending. Root launcher still depends on local source/venv/Python. No executable rebuild or user session termination was performed.

---

# Beginner UI & clustering — 2026-10-26

Implemented the requested clustering task and numeric feature selection, K-Means/DBSCAN/Hierarchical settings, scaled/median-imputed exploration, PCA group display, cluster profiles, same-feature method comparison, K-Means count scan and grouped CSV export. Comparisons display coverage/noise and explicitly avoid prediction-accuracy or universal-winner claims. Learned transforms here are exploratory whole-dataset transforms, separate from supervised validation.

Overview's quality metric opens actual issue details and links to the selected analyzer issue. Its large duplicate visualization section is removed; Visual Analytics retains histogram/scatter/bar charts. Simplified headings and optional technical explanations are applied across the IDE. Analyzer issues are moved above long guidance. Background/card separation and text/control contrast are increased.

Verification: 43 tests pass. Native checks passed: examples/beginner-final-check/result.json, examples/beginner-explorer-regression/result.json, examples/beginner-research-regression/result.json. The root EXE SHA256 remains 1240DE193E576BA06A1D716BC71BB3C11966CC03573CAF0F73B693FA742E50A8. No executable rebuild. Clustering remains numeric-only, bounded exploration; categorical clustering, auto-tuning and learned cluster prediction on new datasets are not implemented.

---

# Pastel reference update — 2026-10-26

Primary entry point: project-root MLStudio.exe. This launcher loads local source/venv on every launch; app edits need a restart, not another EXE build. Older dist folders are retained as historical builds. No standalone packaging claim is made for the development launcher.

Implemented reference controls: pastel dashboard with real counters/dataset composition/distributions/activity; automatic histogram/scatter/category bars; complete-dataset search, stable sorting, pagination, visibility/selection; full-column descriptive statistics; rename, strict type conversion, text-based row filtering and undo/redo; transparent health indicator, cleaning-plan preview and before/after quality counts; task guidance, severity/evidence panels and native HTML/CSV/PDF health export; workflow search/categories/auto layout/minimap; Error Detective purpose/evidence/replay explanation; persisted theme/autosave/storage/research-row-budget preferences, CPU capability display and informative status bar.

Verification: 33 pytest tests passed (14.01s); JavaScript syntax checks passed; native reference acceptance passed, including launching through root MLStudio.exe. Native five-feature research acceptance passed and its exported reproduce.py independently verified the results. See examples/launcher-pastel-final/result.json and examples/research-pastel-final/result.json. A requested fresh-context review could not run because the reviewer model returned an account usage-limit error; direct code review and automated/native checks were completed, not represented as an independent review.

Limits: the health indicator measures only completeness and exact duplicates; row selection does not edit cell values. Text row filtering is literal contains, not a general expression engine. Autosave requires an explicitly saved project. CPU only. Existing research resource limits remain; all original master-spec features are not claimed complete. Source data and virtual environment must remain alongside the launcher. Current source/UI is authoritative, not previous packaged snapshots.

---

# Research features update — 0.4.0

All five requested features are implemented with real local operations: Safety Advisor, Preprocessing Impact Lab, Reliability CV comparison, Error Detective and Reproducible Research Pack. See README Research tools for exact scope and resource limits.

Verification so far: full suite 29 passed (13.84 seconds); source research UI acceptance passed all five features, native export and one-use final lock. Exported Python replay has passed in a separate process; altered data is rejected by fingerprint. Group/time isolation, training-only imputation, paired comparisons, rare classes, nullable error slices and transactional config settings are tested. Fresh-context review and final Windows research build acceptance are in progress.

Rulings: native implementation in the existing checkout because the workspace has no Git repository; previous executables and user sessions remain untouched. Classification research uses macro F1 to expose minority-class failures; legacy Model Lab keeps weighted metrics. SD is labeled descriptive, not a confidence interval. Final-test exposure is audited by original source fingerprint; reused/edited related datasets cannot be certified independent. Current study runtime state is not embedded in ordinary project JSON: export the full Research Pack for replay. No research cancellation, automatic hyperparameter tuning or PyTorch is claimed.

---

# Guided CSV update — 0.3.0

The normal import-options popup is removed. Advanced options remain opt-in. CSV import returns to an automatic visual overview with real histogram, scatter, and category bar charts (at most 2,000 sampled rows, explicitly labeled). Smart Analyzer includes evidence-based preprocessing steps and previews. Model Lab includes task/target selection, per-algorithm rationale and dataset-specific cautions. Model comparisons explain observed leaders, ties, metric direction, score gaps and validation limits.

Verification: 18 tests passed (latest run 6.49 seconds), including no default import popup, no guessed target, visual data summaries, preprocessing suggestions and tied comparisons. Source embedded-UI acceptance passed for automatic charts, actionable preprocessing, real training/comparison explanations, workflow and charts. Final guided executable build succeeded and native packaged-UI acceptance passed with exit code 0. Result: examples/guided-packaged-final/result.json. All eleven bundled Python modules and three web assets were verified to match source. Latest full test run: 18 passed in 5.58 seconds.

New distribution folder: dist/MLStudio-Guided. Prior builds and user sessions are retained. The modern shortcut will point to the verified guided build.

Remaining limits: automatic import assumes UTF-8/BOM, a header row and decimal point unless advanced options are chosen. Suggested algorithms use deterministic rules; performance leadership is an exploratory holdout result, not an unbiased final model-selection claim. CPU PyTorch, configurable CV/tuning, persistent models and installer/clean-machine acceptance remain pending.

---

# Development status

Date: 2026-10-25 (Asia/Dhaka). Version: **0.2.0**. Development release; the full master specification remains incomplete.

## Current interface

At the user's request, the existing EXE now hosts a modern, entirely local web interface using Qt WebEngine and QWebChannel. The Python engine, native CSV/project/export dialogs, sklearn pipelines, transformations, SQLite history and reports remain in place. No hosted Site, HTTP server, remote font, CDN, account or API is required.

Implemented UI:

- Custom sidebar, workspace dashboard, live counters, real dataset distribution and recent experiments.
- Searchable 100-row data preview, column-profile cards and transformation history.
- Analyzer findings and severity filter, evidence/recommendation panels, before/after modal with highlighted changed cells, apply and dataset undo/redo.
- Chart studio displaying actual Matplotlib figures.
- Model setup, metric cards, confusion-matrix heatmap, comparison table, history and inspectable experiment details.
- Drag-and-drop node library, draggable nodes, typed port connections, node settings, deletion/disconnection, graph undo, zoom/fit, validation and execution status. Available nodes: Dataset, Remove duplicates, Train model.
- Session predictions, export cards, project save/open, privacy/settings and keyboard shortcuts. Deep Learning remains explicitly unavailable.
- Local assets and a restricted command bridge. Renderer requests to external URLs and unrelated local files are blocked; untrusted cell contents are escaped.

The native widget controller remains in app.py, while desktop.py and web/ implement the new presentation. This keeps the tested processing engine intact while improving the interface.

## Verification

- **14 automated tests passed in 4.79 seconds.** Coverage includes existing core/data/workflow/training/persistence tests, bridge preview and command allowlist, empty-numeric dashboard handling and external/private-file URL blocking.
- JavaScript syntax checked with Node --check.
- Native embedded-UI acceptance passed: WebEngine startup, local assets, QWebChannel, preview/apply/undo, real workflow creation, actual model training and metrics, six-model comparison, workflow execution and Matplotlib chart display. Artifacts: examples/modern-acceptance/; all test data is explicitly synthetic and history is isolated.
- Dashboard, analyzer, preview modal, workflow canvas, model results and comparison screenshots visually inspected. Sidebar was adjusted for laptop viewport heights and Windows scaling.
- The 0.2.0 PyInstaller spec includes WebEngine resources and local assets. Final packaged-UI acceptance passed with exit code 0: WebEngine startup, offline assets, bridge handshake, preview/apply/undo, workflow creation/execution, model training, six-model comparison and SVG charts. Results: examples/modern-packaged-final/result.json. Package: dist/MLStudio-Modern/MLStudio.exe (737.9 MB including embedded Chromium). A root-folder shortcut ML Studio - Modern.lnk points to it. The old 0.1.0 build is retained separately.

## Packaging history

The 0.1.0 package initially failed because PyInstaller picked an incompatible Poppler icuuc.dll from the development environment's PATH. The spec excludes that unrelated ICU and uses Windows' ICU API. The corrected 0.1.0 native package passed startup/training/chart/report smoke tests. This exclusion remains in 0.2.0.

The build emits an optional scipy.special._cdflib hook warning. Exercised scientific-library paths pass, but broader clean-machine validation remains pending. Inno Setup is not installed, so the installer script has not been compiled or installation-tested. Physical network-disconnection and clean Windows VM acceptance remain pending.

## Remaining product requirements

1. Full workflow node catalog, per-node progress/error details, cancellation, caching and redo. The new editor implements only the three real nodes listed above.
2. Target/task-aware quality detectors, class suitability inside Analyzer, date/domain rules, selected-key duplicates and custom import missing markers.
3. Configurable hyperparameters/CV, sortable comparisons, persistent model bundles and model selection.
4. Optional CPU PyTorch lab, training curves, architecture controls and safe weight export.
5. Richer reports with embedded charts, PDF export, portable project bundles, recent-project indexing, autosave/recovery.
6. Move remaining expensive preview/chart/prediction/profile operations off the UI thread and improve memory limits/streaming. Dataset undo still stores full in-memory versions.
7. Compile and test the Inno Setup installer, then perform clean-machine and disconnected-network acceptance.

Exact next product step: finish per-node execution feedback/cancellation and extend the available workflow nodes without weakening training-only preprocessing.


# Minimal workflow update — 26 October 2026

Dedicated Preprocessing now separates evidence-based manual cleaning from training-only preparation. Outlier/missing/duplicate/label findings open right-side exact affected-row previews with detection evidence; change previews and Apply stay inline, and Undo/Redo preserve the source. Mean/median filling, standard/robust/no scaling, training-only IQR capping and explicit feature exclusions are persisted with projects and used by simple/workflow training. Whole-dataset fitted imputation/IQR edits are rejected before evaluation until undone. Advanced research retains its own declared preparation.

The main sidebar has 7 destinations; all available specialist routes remain accessible through Advanced and Results. Guided Train requires a task/outcome and uses a clear starting algorithm, with optional 3-method comparison. Dataset details and specialist controls are folded. Visual requests now include sorted numeric/ISO-date trends with separate series, category percentage shares and separately reviewed question queues.

Verification: 91 automated tests passed; native acceptance in examples/minimal-verified exercises outliers beyond the first page, inline preview/apply/undo, leakage guard, actual training and 3-method comparison, all 18 available routes, multi-question trend/share results, Light/Dark and original-data preservation. Data shown in acceptance captures is explicitly synthetic QA data, not benchmark claims. See docs/PRODUCT_VALUE.md for the product explanation, demonstration and limitations. Existing launcher/source environment and incomplete standalone installer/model persistence limitations remain.

## Expanded training preparation — 26 October 2026

Training now supports five numeric imputation strategies, configurable category filling, seven numeric scale choices including Min–Max and L1/L2 normalization, missing flags, discretization, constant feature removal, F-test selection and PCA. All learned steps remain inside the training split. The supervised registry contains 22 methods (10 classification, 12 regression), including baseline references. Extra controls remain folded and trained results show actual prepared feature evidence. Full suite: 141 passed. See docs/FEATURE_COVERAGE.md for usage, exact scope and remaining Orange/Weka gaps. Root launcher unchanged.

## Orange/Weka audit repairs — 26 October 2026

The task selector, unsaved preparation handoff and small-KNN failure are repaired. Saved preparation is shared with fold-local Advanced evaluation; fitted models can be saved/opened using versioned skops bundles; prediction CSV options, user-selected comparisons, bounded model settings/search, linked scatter selection, two useful workflow nodes and embedded project snapshots are implemented. Full suite: 150 passed. Native source and rebuilt frozen EXE acceptance both pass all nine repair flows. Details and limits: docs/AUDIT_FIXES.md. The root launcher stays unchanged. The portable folder at dist/MLStudio-Research includes its runtime; a clean Windows VM remains an external validation task.

## Quiet Home page — 26 October 2026

Home now uses a flat layout with separators instead of repeated statistic cards, workflow boxes and promotional panels. It retains actual dataset/version/duplicate information, row/column/missing/finding counts, a contextual next step, four working navigation links, recent real experiments and model status. Empty states are explicit. Detailed tools remain accessible. New styles are scoped to Home and reuse Light/Dark/High contrast colors.

Native visual/navigation acceptance passes in examples/home-quiet-verified/result.json, including a compact window without horizontal overflow and unchanged data. Source and portable index.html/home.css/home.js hashes match. This UI-only update does not rebuild or replace either executable; reopening loads the changed assets.

## Student dashboard and evidence upgrade — 26 October 2026

Home now uses one focused dataset surface with real counts, stronger visual hierarchy and a restrained accent rather than the preceding all-flat layout. Ordinary training results include holdout source rows, clickable confusion-cell filtering, a simple baseline, balanced metrics, binary ROC and optional held-out permutation importance. A new prediction workspace supports input/mapping review, class probabilities, preview/export and single-record inference. Saved-model-only prediction and exact project numeric snapshot roundtrip bugs are fixed. Full suite: 155 passed; refreshed contest audit: 46 checks passed in both source and rebuilt frozen EXE. Native Home, new prediction/evaluation, dataset visualization, visual request and research smoke flows pass; the new student/evaluation flow also passes in the frozen runtime. Root launcher remains unchanged. See docs/STUDENT_UPGRADE.md for exact coverage and remaining gaps.

## Reactive canvas and main Visuals — 26 October 2026

Visuals is a main sidebar destination directly below My dataset and combines chart building/questions. Workflow is a main destination with a 16-step searchable library, free-position round nodes, typed drag connections, animated arrows, automatic background computation, branch-local errors, cached outputs and click-to-open settings/output. CSV/table/columns, branch cleaning and deferred preparation, all five chart types, training/comparison/results and training-only prepared output are implemented. Node parameters and incomplete canvases survive project reopen. Full suite: 167 passed; native mouse linking/movement, real preparation/model outputs/export, and existing question/export flows pass. Rebuilt portable EXE passes all nine reactive acceptance checks, including outcome-free preparation preview and subsequent training/results. Root launcher remains unchanged. See docs/REACTIVE_WORKFLOW.md for actual scope and limits.

