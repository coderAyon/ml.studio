# Business analysis workspace — 29 October 2026

Business mode is a real offline, single-table BI workspace inside the existing Qt WebEngine application. Choose **Business analysis** under Workspace mode in the sidebar. Choose **Machine learning** to return to the preceding ML page without replacing the dataset, graph or fitted model. Fresh preferences default to ML; the selected mode persists locally. Dataset name, row/column counts and its remove action now appear immediately below the ML Studio brand in both modes. Removing the workspace dataset retains the source file and uses the existing confirmation.

## Use it

1. Import CSV/dense ARFF with the existing importer, or open a project.
2. Enable Business analysis and open **Report setup**.
3. Choose the measurement and calculation. Count means all matching records; other choices include sum, mean, median, min, max and distinct nonmissing values. Select segment/date/pivot/customer fields when useful. Business meaning is explicitly assigned by the user; the software does not assume an ID is revenue or guess currency.
4. Add optional AND filters/date range, then **Build business report**. Every page shares that filter context. Clicking a segment filters the report; remove chips or Clear filters to restore it.
5. Review evidence or export. **Save project** includes the original typed dataset snapshot, approved cleaning log and report configuration. **Save report definition** stores configuration only, and opening it recalculates against the active dataset. Calculated outputs are invalidated when the dataset revision changes.

## Implemented coverage

| Area | Actual behavior |
|---|---|
| Mode and navigation | Eight business routes: Overview, Report setup, Segments & Pareto, Time trends, Pivot table, Targets & what-if, Drill-down, Reports & exports. Existing dataset/preparation/charts/workflow tools remain linked; ML mode restores the existing navigation. |
| Data connection and preparation | Shared CSV options/dense ARFF/project import; existing column profiles, missing/outlier evidence, preview/apply/undo/redo and source protection. Independent table joins/concatenation are available through existing workflow nodes. Business reports analyze the active workspace table, not an implicit relational model. |
| Measures/KPIs | Count, sum, mean, median, min, max, distinct; filtered record count, observed/finite measurement rows, missing/infinite counts, selected customer/entity count, explicitly mapped paired revenue/cost/profit/margin. No executable expression evaluation. |
| Filters | Up to eight AND conditions: equality/inequality, literal case-insensitive contains, numeric >/>=/</<=, missing/present, plus optional inclusive UTC calendar date bounds. Null group drill-down uses missing semantics; a literal `(missing)` group remains distinct. |
| Segmentation/Pareto | Descending actual segment values; configurable top 1–30, record counts, shares and cumulative percentages. Bar click cross-filters the report. Negative bars use a centered zero baseline. Pareto is defined only for nonnegative additive totals/counts; all groups contribute to its denominator even if only top N is shown. |
| Time intelligence | Day/week/month/quarter/year UTC buckets, actual aggregated values and record counts, missing/unparseable date count, latest versus previous **observed** period change. Calendar gaps retain their spacing and are not treated as zero; partial periods are explicitly disclosed. |
| Pivot | Two explicit dimensions and selected aggregation; preserves missing groups, distinguishes undefined numeric cells from absent count cells. Up to 100 row groups / 30 heading groups; distinct counts are not advertised as additive totals. |
| Targets/scenario | Target value, gap, attainment for positive targets, arithmetic -100% to +1000% sensitivity to selected metric. Zero denominators and all-missing measures stay undefined. A scenario is not a forecast, calibrated confidence interval, causal intervention or promised business outcome. |
| Drill-down | Up to 200 filtered source records and 30 columns in preview, with active row positions. Full filtered CSV preserves every selected row and all columns within the business input bounds. |
| Sharing/replay | Offline escaped HTML with settings/KPI/evidence tables; KPI, segment, trend, pivot and full filtered-data CSV. Atomic destination writes and existing protected-path rules. Definition JSON and ordinary project configuration roundtrip. No data is uploaded. |
| Quality and UX | Real missing/exact-duplicate/infinity counts, explicit preparation link, light/dark and responsive layouts, local background calculation. Field names/category values are escaped in UI and HTML export. |

## Calculation contract and limits

- Up to **100,000 records / 300 columns** per business report. Maximum 500 segment groups, 1,000 observed time buckets, eight AND conditions, top 30 displayed groups and bounded previews. These are local processing safeguards, not a streaming or enterprise-size claim.
- Numeric aggregations exclude missing/infinite measurements. All-missing results remain undefined. Analytical floating-point aggregation avoids integer sum wraparound, but is **not decimal-exact bookkeeping**. Exported underlying data is separate from formatted KPI values.
- Profit/margin use only rows where both explicitly mapped revenue and cost are finite. Margin uses paired revenue as denominator; zero paired revenue returns undefined. Refunds/negative amounts are retained, not silently deleted.
- Distinct customer counts exclude missing labels. Count always counts matching records, including those with missing measurements. Filtering never cleans, imputes or versions the underlying dataset.
- Dates must be explicit text/datetime values; ambiguous numeric timestamps are rejected. Invalid/missing dates are excluded from trends and counted. Numeric KPIs without date-range filtering still include those rows. Trend comparisons use observed buckets and may compare partial periods.
- All report results are tied to the immutable active dataset revision. After cleaning, undo/replacement edits or reopening a project, build a fresh report before exporting. Switching display mode alone does not invalidate data or alter ML state.
- HTML/CSV exports contain private source values and are saved only to the chosen local destination. CSV content is raw data, not a formula execution environment; handle it appropriately when opening untrusted source values in spreadsheet software.

## Research and deliberate remaining differences

Official documentation informed the workflow rather than establishing universal tool parity:

- [Microsoft Power BI overview](https://learn.microsoft.com/en-us/power-bi/fundamentals/power-bi-overview) describes data connection, visual reporting, sharing, goals and governance across desktop/cloud components.
- [Power BI measures](https://learn.microsoft.com/en-us/power-bi/transform-model/desktop-measures) explains dynamic aggregations and report/model measures. This implementation supplies an explicit bounded aggregation selector; it does not implement DAX's expression language or relational evaluation semantics.
- [Tableau filter actions](https://help.tableau.com/current/pro/desktop/en-us/actions_filter.htm) informs shared filter context and mark-to-detail exploration. This mode implements segment-click filtering within the active table, not arbitrary cross-source sheet actions.
- [Tableau forecasting](https://help.tableau.com/current/pro/desktop/en-us/forecasting.htm) distinguishes actual forecasting models and constraints. ML Studio's business time trends and what-if values are explicitly descriptive/arithmetic; no unimplemented forecast model is shown as a prediction.

“A to Z” enterprise BI is not claimed complete. Remaining capabilities include SQL/Excel/warehouse/live connectors, semantic multi-table models and arbitrary DAX measures, free-layout dashboard authoring, scheduled refresh, cloud collaboration/publication, roles/row-level security/audit governance, geography/maps, validated seasonal forecasting and uncertainty, cohort/RFM/funnel/retention-specific models, and dashboard PDF/PPT/image composition. The existing ML engine/charts are accessible for additional analysis, but that does not make these business-specific workflows implemented. Future increments should add measured workflows with tests rather than placeholder menu entries.

## Verification

Production-function tests cover actual totals, paired profit, missing/negative/infinite values, zero denominators, customer counts, filter scope/source positions, missing group labels, calendar gaps, integer sum overflow, invalid settings, export/definition roundtrip and revision invalidation. Native `packaging/verify_business.py` exercises real form controls, mode switching/restoration, clickable segments, all six export types, project reopening, sidebar removal and 48 page/size/theme layouts. The existing root EXE comparison acceptance also runs to check ML behavior after the shell change. Evidence and synthetic fixtures: `examples/business-verified-2026-10-29/`. Synthetic data does not establish business performance or usability superiority.
