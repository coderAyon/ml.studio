# Research features design

Implements the five features explicitly requested after the competitive-positioning discussion. Existing offline desktop, CSV privacy, original preservation and training-only learned preprocessing remain mandatory.

1. Safety Advisor: evidence-based target suitability, repeated observations, possible subject/time columns, target copies, high correlation and protocol recommendations. No heuristic proves leakage; user chooses group/time columns.
2. Impact Lab: compare median/scaled baseline against mean imputation, no scaling and missing indicators on identical development folds. Fit preprocessing per fold. Show paired differences, runtime and caveats.
3. Reliability: 3-fold x 2-repeat classification/regression CV on development rows; group and time alternatives. Reserve 20% final rows before CV. Fold mean/SD, actual scores, timings, simple baseline and paired differences. No significance claims or automatic final winner. User may evaluate one candidate once on final test; further selection invalidates that estimate.
4. Error Detective: selected final-evaluated candidate only, actual row positions, actual/predicted, classification error or regression absolute error, class/missingness/numeric/category slices with counts and explicit small-group caveats. Original feature rows viewable.
5. Research Pack: local ZIP with active dataset, SHA-256, selected features/protocol/seed, exact positional splits, configurations, observed results, errors, environment versions, transformations, original provenance and runnable Python replay plus module sources. Data inclusion is disclosed in the export UI; no remote transmission. Verify replay against original metrics.

Resource limits: <=20,000 rows for new repeated-CV tools and <=200 input columns; bounded one-hot categories, sequential CPU fits; background execution. Keep old labs working. Clear stale research results on dataset version, target or validation changes. Retain previous EXEs and build a new research distribution.
