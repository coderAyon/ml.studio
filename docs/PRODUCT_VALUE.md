# ML Studio: what the current product offers

## A defensible contest explanation

ML Studio is an offline workspace for people who want to understand a CSV and run a first machine-learning experiment without assembling a large workflow. Its main path is Dataset → Preprocessing → Visualize → Train / compare. The difference to demonstrate is the guided experience and visible evidence behind decisions, rather than a claim to have more algorithms than every existing product.

1. **Review the exact affected records.** Clicking an outlier, missing-value or duplicate finding opens matching rows on the right. IQR evidence includes finite-value counts and the actual limits. A finding is not silently treated as an error.
2. **See a change before applying it.** Previews report changed cells, removed rows, missing counts and before/after values. Applying creates an undoable dataset version while preserving the source CSV.
3. **Keep learned preparation inside training.** The simple training path fits numeric filling, category encoding, scaling and optional IQR capping on training rows. Mean/median, standard/robust/no scaling and explicit feature exclusions can be chosen once in Preprocessing. Fitted whole-dataset imputation/outlier edits must be undone before evaluation.
4. **Turn supported questions into reviewable visual plans.** The offline rule matcher suggests actual CSV columns, then the user reviews the calculation before execution. Group summaries, distributions, relationships, spread, correlation, missingness, ordered trends and percentage shares are supported. Up to six questions can be prepared independently.
5. **Show the evidence behind every result.** Visual outputs expose their tables, group sizes, finite-value counts, filters and sampling rules. Simple training shows real holdout metrics; three-method comparison uses identical splits. Advanced tools offer repeated validation, reserved final evaluation, error inspection and replayable research exports.

These are current implementation claims, not a verified exclusivity comparison with Orange or Weka. To claim faster/easier usability, conduct the same tasks with novice users and measure completion time, errors and assistance required.

## Suggested five-minute demonstration

- Import a CSV containing a documented outlier and a missing feature value.
- Open Preprocessing, click the outlier and point to the exact affected row and limits.
- Preview a change, cancel or apply, then Undo. Show the original CSV is preserved.
- Choose training-only outlier capping and exclude an identifier. Show that the active dataset did not change.
- Ask a group-summary question and a time-trend/share question. Review the plans and export the actual tables/reports.
- Choose an outcome and Train. Compare three starting methods; describe the metrics as exploratory holdout estimates.
- Open Advanced tools only if asked about stronger validation or reproducibility.

## Boundaries to explain honestly

- Natural-language matching uses a bounded offline grammar and actual column names; it is not a general-purpose LLM.
- An outlier is a heuristic observation, not proof of incorrect data. Numeric training IQR zero leaves that feature unchanged.
- Simple Train uses a deterministic 75/25 split; grouped/time data require an appropriate design in Advanced tools. Advanced research has its own declared preprocessing settings.
- Session models and visual boards are not saved as fitted models by ordinary project files. Export reports before closing.
- Deep Learning/GPU and a clean-machine standalone installer remain unfinished. The project-root EXE is a stable launcher for the current local source and environment.

Verification evidence for the minimal flow is in `examples/minimal-verified/result.json`; all values in its UI captures come from explicitly synthetic QA data, not performance benchmarks.
