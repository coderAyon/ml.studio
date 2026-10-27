# Custom dataset column widths

In My dataset, drag the right edge of a data-column header to resize it from 72–1,000 CSS pixels. Double-click the edge to fit the header and current-page values (bounded at 500 px). A focused resize handle also supports arrow keys, Shift for larger steps, Home and End. Escape cancels an active drag.

Open **Column widths** for an exact pixel width, Compact columns (88 px), Auto-fit visible, or Reset widths (160 px). The Select and Row columns keep their fixed widths and remain pinned during horizontal scrolling. Long values can be wrapped with the existing checkbox or inspected through their title/row details. Resizing is a view preference and changes no CSV values, training features or model state.

Widths stay in the current dataset session across sorting, filtering, pagination, hiding/showing columns and other UI renders. They reset on importing a new dataset instance or restarting the application; widths are not saved into project files. Auto-fit measures up to 100 displayed rows and is not a scan of the entire dataset.

Embedded desktop verification uses a 245-row, 31-column synthetic dataset. Ten checks passed, including a native mouse drag changing the actual rendered width from 160 to 250 px without sorting, preserved widths, exact numeric controls, compact/fit/reset, retained horizontal scroll, Light/Dark views and unchanged original data. Evidence: `examples/column-sizing-verified/result.json`; run with `PYTHONPATH=src` and `python -m mlstudio.column_sizing_smoke`.

Reopen the project-root `MLStudio.exe` to load the updated UI.
