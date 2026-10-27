# Visible outlier review in My dataset

My dataset now reports possible IQR outlier **cells** and distinct affected **rows** across the complete active dataset. Numeric column information includes the finite-value IQR bounds, exact count and up to three most extreme example values with active row positions. Column tips no longer claim that there are no basic concerns when IQR outliers exist.

Flagged table cells show their value with an amber **Outlier** label and flag icon. Click a cell to see its active row, column, value, Q1/Q3 and lower/upper limits. **Show flagged rows** filters the table; column summaries filter to that column's outliers. Search, sort and pagination preserve the filter and original active positions, including datasets with duplicate or nonnumeric index labels. **Show all rows** clears the outlier filter.

The selected cell links to its column's boxplot and the matching Prepare data finding. Prepare data focuses the selected active row instead of dropping it from a bounded preview. No value is changed by viewing, filtering or plotting.

Boxplots label up to two extreme IQR values per column. Scatterplots label up to three extreme plotted flagged points with active row positions. When scatter data exceeds 5,000 complete finite pairs, up to 50 extreme flagged observations are reserved before the remaining points are sampled. Counts still cover all active rows. Turning off outlier markings also turns off these annotations.

Detection uses finite numeric values outside Q1 − 1.5 × IQR or Q3 + 1.5 × IQR. Missing/infinite values are excluded; Boolean/category columns do not receive numeric IQR flags. An unusual value is a review finding, not proof of an error. Scaling does not automatically make an outlier valid or remove it.

Validation: **179 automated tests passed**, including exact bounds/examples, all-page markers, sort/search/filter/cleaning position preservation, annotations and large scatter sampling. The embedded desktop smoke reproduces Iris row 59 with petal length 200 and passes eight checks covering visible highlighting, exact evidence, filtering, boxplot labeling, Prepare data navigation, Light/Dark display and unchanged source values. Evidence: `examples/outlier-review-verified/`; run with `PYTHONPATH=src` and `python -m mlstudio.outlier_review_smoke`.

Close and reopen the project-root `MLStudio.exe` to load the changes. Historical frozen builds under `dist/` do not update automatically.
