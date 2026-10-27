# Compact, user-built workflows

**Latest update:** [Startup and workflow expansion](STARTUP_AND_WORKFLOW_EXPANSION.md) adds explicit independent CSV sources, 56px circles and 38 steps. Earlier size/catalog/source descriptions below document the previous implementation.
Workflow circles are 72 CSS pixels (previously 104), with names below the icon. Port geometry, arrow endpoints and drop centering use the new size. Existing saved graphs keep their nodes and positions. **New workflow** clears the canvas explicitly; Undo restores the previous graph. Importing a dataset does not add steps. The four-node starter action is removed from the current canvas.

Drag any of the 16 steps from the library. Dropping a step leaves the output inspector closed so the canvas stays free; click a node to inspect its settings and results. Click-to-add remains available as an alternative to dragging.

Drag an output circle into an empty canvas area and release to open **What comes next?** A temporary arrow indicates the pending connection. No graph change occurs until a suggestion is selected. Selection adds exactly one new node and one compatible connection in a single undoable edit. Direct port-to-port dragging remains supported. Escape, the close button or clicking outside dismisses suggestions.

Recommendations use explicit offline rules, not a remote model: actual branch missing-cell and duplicate-row counts, numeric/category availability and ancestor steps. Missing-value filling and duplicate review are prioritized when relevant; Prepared output is prioritized after preprocessing; model output offers Results. Each choice explains why it may help. These suggestions are starting points, not a claim that one modeling method is optimal. All compatible catalog steps remain available in the searchable menu.

Validation: 175 automated tests passed. `python -m mlstudio.flow_suggestions_smoke` (with `PYTHONPATH=src`) checks 10 desktop scenarios, including native mouse arrow dragging, native suggestion selection, cancellation without graph changes, 72px dimensions, no automatic inspector, explicit empty workflow and Undo, branch-aware ranking and Light/Dark rendering. Evidence: `examples/flow-suggestions-verified/`. The existing reactive flow acceptance also passed nine checks covering real import, native port connection, node movement, prepared output, export, training, model activation, project saving and consolidated Visuals.

Reopen the project-root `MLStudio.exe` to use the updated source. Saved project graphs are preserved; use New workflow if you want a blank canvas.
