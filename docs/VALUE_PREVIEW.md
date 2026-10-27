# Before / after numeric preparation

Prepare data now has **Preview Before → After on the right** beside the training preparation controls. It previews the current draft, including missing-value filling, IQR treatment, discretization and scaling, without changing the source CSV or saving the draft. After changing a choice, refresh the preview. Save the training plan separately to use those choices for training.

The right-side table keeps the same 1-based active row position, retained index label, input column, original value and prepared numeric value. Original IQR flags remain visible after scaling; normalization changes units and does not necessarily eliminate outliers. Outliers appear first in the bounded default preview. Choose any active row and numeric input column to inspect values outside the default 30-row / 20-column preview.

Normalize, Missing values, Preprocess and Prepared output workflow steps show this comparison directly when clicked. A numeric row/column selection updates the node preview. Workflow packets still carry original branch data and a preparation plan; the display-only transformation is not reused to fit downstream model evaluation.

The preview fits statistics on a deterministic 75% training partition and can transform either training or holdout records, labeled in the table. Min–Max holdout values can exceed 0–1. The table shows the numeric preparation stage before categorical encoding, feature selection or PCA. Prepared output retains its separate encoded training-values table. Original IQR flags describe the incoming dataset and are not model predictions or confirmed errors.

Validation: 175 automated tests passed. `python -m mlstudio.value_preview_smoke` (with `PYTHONPATH=src`) additionally checks the embedded desktop UI: row 3 changes from 100 to 1 under Min–Max, retains its original outlier flag, appears in both Prepare data and Normalize output, and leaves the dataset unchanged. Evidence is in `examples/value-preview-verified/`.

Close and reopen the project-root `MLStudio.exe` to load current source. Historical frozen distributions under `dist/` do not automatically update.
