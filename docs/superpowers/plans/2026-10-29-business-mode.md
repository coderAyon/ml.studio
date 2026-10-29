# Business Mode Implementation Record

**Goal:** Add an offline business-analysis workspace alongside ML mode and move dataset identity/removal beneath the sidebar brand.

**Architecture:** Existing Python/Qt WebEngine/JavaScript/scikit-learn/SQLite retained. A pure descriptive engine and small coordinator share immutable dataset revisions, the worker pool, import/cleaning tools, native dialogs and protected exports. No new runtime dependency or paid service.

**Scope/spec:** [Business-mode usage, calculation contract and coverage](../../BUSINESS_MODE.md). Official Power BI/Tableau documentation reviewed before implementation; enterprise parity and unsupported prediction claims are excluded.

- [x] Inspect bridge, navigation/render pipeline, dataset lifecycle, project formats and existing tests.
- [x] Research primary BI sources; define explicit measures, shared filters, segment/trend/pivot/target/drill-down/export workflows and document missing enterprise capabilities.
- [x] Add meaningful tests first; observe missing-engine failure.
- [x] Implement `business.py` calculations and safe configuration validation; never mutate input records or infer business meaning from a column name.
- [x] Implement revision-safe `business_controller.py`, persisted mode/configuration, local definitions and protected atomic exports.
- [x] Implement separate mode/navigation/pages and shared-tool links in `business.js/business.css`; preserve existing ML workspace state.
- [x] Move existing dataset remove action below the brand beside the bold dataset name, retaining confirmation and source preservation.
- [x] Test actual form interactions, filtering, target/scenario, all exports, project reopen, mode restoration, sidebar redraw/delete and source bytes; inspect screenshots in light/dark and three window sizes.
- [x] Run core/targeted/full regression suites and the existing root-EXE ML acceptance.
- [x] Update README and DEVELOPMENT_STATUS with verified coverage and open work.

**Boundaries:** Single active table; UTC descriptive time buckets; floating-point analytics; explicit sum/paired-cost profit; no automatic forecast or arbitrary code/DAX. Mode is a local preference; report configuration is saved in the ordinary project, while computed outputs are rebuilt after revision changes. Results and exports use a common filter context. No broad architecture rewrite or new standalone package is claimed.
