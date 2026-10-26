# ML Studio functional audit — 26 October 2026

The implemented desktop features were exercised against the actual local Python/Qt application. This is a functional audit on this Windows workstation, not a certification that the entire original master specification or contest deployment is complete.

## Results

- 50 automated tests passed, including failure recovery, numeric/encoding edge cases, training isolation, research replay and source-file protection.
- 35 core end-to-end checks passed through the unchanged project-root MLStudio.exe launcher: `examples/contest-launcher-verified/result.json`.
- Clustering/popup/beginner UI acceptance passed: `examples/contest-clustering-final/result.json`.
- The five research features passed their native acceptance: `examples/contest-research-verified/result.json`.
- JavaScript syntax checks passed. The test suites use generated QA datasets; real sklearn fits, transformations, charts, predictions and file exports run. Native file/option dialog selections are automated, rather than requiring a person to choose each test path. No fabricated benchmark scores or demo results are inserted into the normal app.

## Functional coverage

| Area | What was checked | Result |
|---|---|---|
| Navigation | All 15 available pages render; unavailable Deep Learning is explicitly disabled | Pass |
| CSV import | Normal CSV; no guessed target; advanced Latin-1, semicolon and comma-decimal parsing | Pass |
| Dataset Explorer | Entire-dataset search including the last row, stable sorting, pagination, visibility, row selection and descriptive statistics | Pass |
| Cleaning | All 11 exposed transformations, previews, apply, undo/redo, bulk preview and version/source preservation | Pass |
| Quality checks | Real issue popup, severity/evidence, direct analyzer link, health formula disclosure and report exports | Pass |
| Classification | Logistic Regression, Random Forest, Decision Tree, KNN, SVM and Naive Bayes fit real pipelines and display holdout metrics | Pass |
| Regression | Linear Regression, Ridge, Random Forest Regressor, Decision Tree Regressor and SVR | Pass |
| Comparison | Six classification methods and five regression methods on consistent holdout splits | Pass |
| Prediction | New compatible CSV, exact feature-schema validation, actual exported predictions and colliding output-column protection | Pass |
| Charts | Histogram, box plot, scatter, correlation, missing values and category frequencies produce actual Matplotlib SVGs; numeric chart controls select numeric columns | Pass |
| Clustering | Numeric feature selection, three methods, configurable settings, same-feature comparison, coverage/noise, group plot/profile, K scan, label CSV and stale-result invalidation | Pass |
| Workflow | Starter pipeline, actual model execution, empty/disconnected validation errors, deletion, graph undo, zoom/fit, search, layout and minimap | Pass |
| Project | Save/open roundtrip, source fingerprint verification, transformations/graph/target restoration and autosave | Pass |
| Exports | Cleaned CSV, transformation JSON, experiment HTML and health HTML/CSV/PDF output | Pass |
| Settings | Theme, autosave, row-budget validation and next-launch local-storage selection | Pass |
| History | Actual local SQLite records and inspection modal | Pass |
| Research | Safety evidence; grouped/chronological isolation; repeated shared-fold CV; paired preprocessing study; explicit one-use final test; error rows/slices; exported replay/fingerprint validation | Pass |
| Error recovery | Rare classes, bad task/target, changed source, missing/invalid connections, empty cleaning plans, infinity and memory-budget rejection | Pass |

## Bugs fixed during the audit

1. Background errors could open a warning on the hidden legacy window. The embedded interface now receives errors and remains usable; a valid run immediately after a failed run is tested.
2. Empty workflows were reported as valid. Validation/execution now requires a dataset node.
3. Cleaning could remove every row/column and break the interface. Such versions are rejected; multi-step cleaning is validated before applying any step.
4. Infinite values could break the overview distribution. They are inspectable, flagged explicitly and correctable with an undoable infinity-to-missing transformation. Training rejects uncorrected infinity.
5. Exports could overwrite a source, saved project or experiment store if a conflicting path was chosen. Output paths now protect those files.
6. A project save's fixed temporary filename could collide with another file. Saves now use unique temporary names. Research ZIP exports also use unique temporary names.
7. Prediction export could overwrite an input feature named `prediction` in the exported copy. The output label now gets a unique column name.
8. A renamed column with surrounding spaces could collide with an existing name after trimming. Normalized names are validated before applying.
9. UI bootstrap could render before all interface scripts were loaded. QWebChannel now starts after DOMContentLoaded, so task choices and layout overrides are installed before first render.
10. Unsupported training shapes/tasks and high-cardinality dense encodings could fail unclearly or allocate excessive memory. They now fail with actionable messages before fitting/allocation. Model Lab supports 8–20,000 rows, up to 200 input features and a 5-million-cell encoded budget.
11. A decorative workspace arrow suggested a menu that did not exist. It is now labeled LOCAL. Deep Learning is marked unavailable instead of appearing as a working option.

## Not implemented or not yet verified

| Item | Current state | Contest implication |
|---|---|---|
| Deep Learning / GPU | Not implemented; navigation disabled and labeled unavailable | Do not present it as a supported feature |
| Trained-model persistence | Predictions use the current session pipeline; ordinary project files do not save fitted model objects | Keep the session open or retrain after reopening; saved model bundles are pending |
| Research-session resume | Project JSON saves dataset/workflow settings, not the live research session; the Research Pack supports replay | Use the Research Pack to preserve/reproduce studies |
| Standalone distribution | Root EXE is a development launcher requiring this project's source, `.venv` and its configured Python installation | Copying only the EXE to a judge's PC is insufficient |
| Installer / clean Windows PC | Not verified in a clean machine/VM | Must be completed before a portable/installer contest submission is called ready |
| Cancellation, tuning, expanded workflow catalog | Not exposed as working controls | Do not claim these capabilities |

All passed checks ran locally with the installed dependency versions. This audit does not establish generalization accuracy, significance, scientific novelty, performance on arbitrary large datasets or correct domain interpretation. Grouping scores are internal exploratory scores; supervised holdout comparisons remain exploratory. Research final-test reuse warnings and split restrictions remain in place.

## Running the audited version

Close and reopen project-root **MLStudio.exe**. It loads the updated source; the executable was not rebuilt. SHA-256 remains `1240DE193E576BA06A1D716BC71BB3C11966CC03573CAF0F73B693FA742E50A8`.

Repeat the core audit from the project directory with:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe main.py --contest-audit examples/new-contest-audit
```

Use a new output folder for acceptance runs. Generated outputs contain test datasets, isolated history and settings; normal user CSV files are not modified.
