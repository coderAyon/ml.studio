# Unified workspace and clear evidence — 27 October 2026

## Interface

Home and My dataset now share one **Workspace** navigation entry. It opens an import/project entry point before data is loaded, and the actual dataset table afterward. The current file, row/column counts, quality facts and preparation action appear in the workspace. Experiment history and saved work stay in **My results**, with no recent-experiment section on Workspace. Existing links to Home route here. Data, Columns and Changes tabs retain the original explorer, column profiles and transformation history.

Column visibility, custom widths, wrapping, sorting, search, row selection, pagination and full preview remain available. Less frequent table controls are grouped under **Table options**. Outlier totals and flagged-cell labels stay visible; detailed column examples/limits expand on demand. The top bar contains dataset removal, offline status and the theme switch. Profile/avatar and duplicate help controls are removed, as is the sidebar's decorative My workspace/LOCAL box. The root breadcrumb displays Workspace once.

Shell cleanup was checked across eight routes with empty/loaded datasets, Light/Dark switching, retained table controls, Settings guide keyboard focus and result-history access. Screenshots and results: `examples/shell-cleanup-verified/`. No dataset values or stored experiments are removed by this presentation change.

### Current design and Settings update

The sidebar entry is now **Settings**. The page contains only working preferences: theme, reduced motion, autosave, storage folder and an expandable evaluation budget. The previous guide entry and capability/limitation descriptions no longer appear there. Input validation and relevant operation feedback remain part of the actual tools.

**More tools** is a six-tool directory for validation design, repeated comparison, preparation impact, prediction errors, research packs and reports. Workflow stays in the main sidebar. The full quality report is reached from **Prepare data**, so neither is repeated in this directory.

`web/modern.css` supplies the shared visual system: a proportionate sidebar, lighter canvas, defined white/dark surfaces, consistent controls, restrained purple accents, readable typography, visible keyboard focus and responsive layouts. Workflow circles remain 56px; table widths/resize handles retain their own geometry.

Desktop verification confirmed that the four preference controls update actual application state, all six specialist routes remain distinct, the quality-report link works and seven main pages fit the minimum 1080×720 window without horizontal page overflow. Light, Dark and High contrast captures are in `examples/modern-interface-verified/`. The column-sizing desktop acceptance passed with this design. Workflow acceptance also passed nine checks including native node/port dragging, automatic computation, actual training/prepared outputs, model activation, export and project persistence; evidence: `examples/modern-flow-verified/`.

Prepare data uses compact finding rows and a useful data-check table instead of the large empty placeholder. Before/after cleaning tables sit side by side when space permits. Counts remain visible; detailed methods, numerical bounds and other settings remain available. Shared panel/header/empty-state styles also simplify the other tools in Light and Dark themes.

## More informative preparation

Numeric previews now include effects per input: missing values before/after, changed-cell count and before/after ranges/means. Effects cover all active input rows, using transformations fitted only on the training partition. Display defaults remain bounded to 20 inputs; choose a numeric column to inspect others. Encoded input counts and train/holdout sizes are included. These are data-effect diagnostics, not a promise of better predictive performance.

**Compare preparation alternatives** compares the current draft with one changed choice: robust/standard/Min–Max scaling, mean/median filling, or training-only IQR capping. Both plans use the same split. Comparing changes neither the CSV nor the saved training plan. **Use this preparation plan** explicitly saves the candidate. Preview/comparison work runs on a background worker, and changing dataset, outcome or task invalidates stale evidence.

## Explain and export a result

Results now leads with the fitted method/outcome, the actual primary score and the same-holdout baseline. It shows training/test row counts and input count. Existing confusion-matrix cells, error records, ROC, input explanation, predictions, model saving/loading, experiment history and other exports remain accessible through focused expandable sections.

**Export result brief** writes a self-contained printable HTML file containing the fitted run's protocol, actual metrics, baseline/improvement, preparation, warnings and dataset fingerprint. It uses model metadata, not whichever dataset happens to be open now. User-provided labels are escaped. Export/project-save guards now also protect every independent workflow CSV source.

## Workflow and startup

The preceding [startup/workflow expansion](STARTUP_AND_WORKFLOW_EXPANSION.md) is included: explicit independent CSV imports, 56px circles, 38 functional steps and single-instance launch feedback. A changed file invalidates only its own cached branch; explicit Update workflow rereads file sources and recomputes. New nodes/connections have brief arrival effects, while running arrows animate. Reduced-motion preferences disable these effects. The animated native splash stays responsive during library imports and fades away after the workspace's first real render; its lifecycle/animation and actual EXE launches have separate acceptance scripts.

## Verification and scope

**236 automated tests passed.** The unified desktop acceptance passes ten groups of checks, including merged navigation, retained table/profile/history controls, compact panels, exact preparation effects, unchanged source/saved plan during comparison, explicit candidate adoption, actual training, baseline presentation, HTML export and Light/Dark rendering. The column-sizing and expanded workflow acceptances were rerun. Evidence and screenshots are in `examples/quiet-workspace-verified/`, with the exported synthetic `result-brief.html` available for review.

The project-root EXE remains unchanged; close/reopen it to load source/UI updates. A contest outcome or superiority in beginner task completion has not been established. Validation with real novice users and clean-machine installer acceptance remain separate work, along with the specialist functionality listed in the earlier audit.
