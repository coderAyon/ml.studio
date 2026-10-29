# ML Studio Correctness Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent stale revision results, lossy project/research data and invalid held-out explanations through three independently reviewable repairs.

**Architecture:** Keep the current Python/Qt WebEngine/QWebChannel/JavaScript/scikit-learn/SQLite structure. Introduce a cheap immutable runtime revision identity, strengthen the existing typed snapshot codec, and share the existing held-out validator. Reuse current worker and bridge interfaces; no frontend framework, model serialization technology or execution architecture replacement.

**Tech Stack:** Existing locked Windows CPython/PySide6, pandas/NumPy, scikit-learn, skops and SQLite; no mandatory network/API or new runtime dependency.

**Spec:** [Engineering audit and user requirements](../../ENGINEERING_AUDIT_2026_10_28.md). Baseline `3f24d03`. User authorized execution with “continue”. Task 1 is verified (285 tests / 13 native checks); Task 2 is also verified (365 tests / 16 native checks); Task 3 remains pending after the user steered work toward interface quality. No broad rewrite is authorized.

## Global Constraints

- Preserve original CSV files; preview, explicit Apply, undo/redo and operation history remain available.
- Fit learned preprocessing only inside training folds; preserve random/stratified, group and chronological protocols.
- Never load untrusted pickle/joblib artifacts; retain skops' fixed application trust policy.
- Keep offline operation and all working features. No unrelated rewrites or broad architectural change without explicit approval.
- Write and observe meaningful failing regression tests before each core change; run relevant tests and the full suite after each increment, then native acceptance where applicable.
- Keep old documented project/research encodings readable; never silently coerce an unsupported value. File hashes are integrity checks, not authenticity signatures.
- Record actual verified status in README and DEVELOPMENT_STATUS after each task, including remaining P0 issues. Do not equate UI presence with completion.
- Task completion requires independent review of its semantic result. Implement these tasks sequentially because they touch shared data/evidence boundaries.

## Review Focus

- Undo followed by a different edit at the same displayed version must never revive results or permit stale export/model activation; Task 1 tests it.
- Default-selected cluster columns change when the outcome changes; an explicitly pinned configuration should retain appropriate cache reuse; Task 1 tests it.
- Boundary integers, null masks, float bits, timestamp precision/timezones and nominal domains must survive persistence; Task 2 tests them.
- Legacy saved formats and missing source files must have a safe, explicit compatibility path; Task 2 tests them.
- Identical value hashes with renamed/type-changed columns or malformed test positions must not produce explanation evidence; Task 3 tests them.

## Task 1: Immutable dataset revision and complete cache context

**Files:**
- Modify `src/mlstudio/data.py`: `DatasetVersions` identity only; keep visible `index` and applied log semantics.
- Modify `src/mlstudio/flow_controller.py`, `comparison_tools.py`, `desktop.py`, `research_controller.py`: consume one revision key for caches and publication/export/activation guards.
- Modify `src/mlstudio/flow_engine.py`: include effective clustering inputs/context in cache signatures.
- Test `tests/test_reactive_flow.py`, `test_expanded_workflow.py`; create `tests/test_revision_identity.py` for lifecycle/association integration.
- Add a branch-history scenario to existing native flow/comparison acceptance; preserve isolated fixture/settings behavior.

**Interfaces:**
- Consumes existing `DatasetVersions.apply/undo/redo`, `FlowEngine.run`, `FlowCoordinator.signature`, `ComparisonTools.key`.
- Produces `DatasetVersions.revision_id: str` read-only property. Store one opaque UUID per retained version; new apply always creates a fresh ID, undo/redo return that version's original ID. Keep runtime identity separate from deterministic content/schema fingerprints used for replay/model validation.
- Existing `FlowEngine.run(..., dataset_key=...)` shape remains; coordinators pass the immutable revision instead of the displayed index. UI `version` remains the human-readable history position.

- [x] Add `test_replacement_branch_has_new_revision`: apply A → capture ID → undo → apply B at same index; assert different IDs. Add `test_undo_redo_restores_revision`: exact prior version returns the same ID. Assert failed/empty apply does not change ID/history and discarded redo IDs are not reused.
- [x] Run `.\.venv\Scripts\python.exe -m pytest -q tests/test_revision_identity.py`; observe failure before introducing the property.
- [x] Add revision IDs alongside stored versions, preserving copy-on-read/source behavior. Do not calculate a full frame hash on every UI refresh.
- [x] Add `test_replacement_branch_invalidates_workflow_and_association_exports`: compute/filter A, publish undo, apply B at index 1; assert new table values, stale rules hidden and stale export/activation rejected. Tests must use real rules/model/table results rather than only a sentinel.
- [x] Run those tests before changing coordinators; observe the existing cache collision. Replace revision-dependent keys consistently, including asynchronous completion key checks. Inspect every `(id(dataset), index)` consumer; keep unrelated configuration keys.
- [x] Add parameterized `test_default_cluster_inputs_change_with_target` for K-Means/DBSCAN/Hierarchical and an explicit-column control. Assert resolved input lists and recomputation, not just a changed hash. Observe failure, then include resolved inputs/effective context in cluster signatures.
- [x] Run targeted suites: `python -m pytest -q tests/test_revision_identity.py tests/test_reactive_flow.py tests/test_expanded_workflow.py tests/test_table_operations.py tests/test_workspace_matrix.py tests/test_research.py` using `.venv\Scripts\python.exe`.
- [x] Run full `python -m pytest -q`. Native acceptance: workflow A → leave page → undo → edit B → return/export; association rules must not revive; ordinary undo/redo and node movement cache reuse still work. Check source bytes unchanged.
- [x] Update audit/status with exact results and open limits (external file identity remains separate work); independent review found no blockers. Changes remain local for combined review; no commit/push is claimed.

## Task 2: Lossless versioned project and research snapshots

**Files:**
- Modify `src/mlstudio/project_data.py`: strengthen existing `pack_frame` / `unpack_frame`.
- Modify `src/mlstudio/research_export.py`: use the same codec for new packs, retain legacy reader; include required codec source in generated replay package.
- Modify `packaging/mlstudio.spec`: include codec in frozen replay sources when used.
- Modify `src/mlstudio/app.py` only if additional snapshot schema dispatch is necessary; keep operation replay/portable-source behavior.
- Create `tests/test_project_fidelity.py`; extend `tests/test_research.py`, `test_audit_fixes.py`, `test_student_upgrade.py`.

**Interfaces:**
- Preserve `pack_frame(frame: pd.DataFrame) -> dict` and `unpack_frame(bundle: dict) -> pd.DataFrame`.
- Produce a new explicit `encoding` version (proposed `typed-v4`) with exact integer values/dtype/null masks, exact supported float values, supported datetime unit/timezone, category domains/order, and whitelisted provenance (`duplicate_headers`). JSON-safe data only; no arbitrary object restoration or attrs execution.
- Consume legacy project encodings through existing readers. Research `manifest.schema=2` identifies the new snapshot payload; replay must dispatch schema 1 to its historical reader and schema 2 to the shared codec, reject unknown schemas clearly. Hash the exact stored bytes.

- [x] Add `test_unsigned_and_nullable_integer_snapshot_roundtrip`: assert dtype, masks and values for `uint64 [0, 2**64-1]`, nullable `UInt64`, `Int64` boundaries and ordinary integers. Existing behavior must fail before implementation.
- [x] Add exact float/NaN/+infinity/-infinity, boolean/text/category domains/order and duplicate-header provenance tests; assert `pd.testing.assert_frame_equal` and expected attrs after roundtrip. Keep source byte hash unchanged.
- [x] Add `test_project_save_reopen_without_source_preserves_unsigned_values`: native controller save → delete only synthetic source → reopen → export; assert the identifier string remains `18446744073709551615`, not `-1`. Include applied clean/undo/redo state and unaffected feature schema.
- [x] Run the new tests and observe the failures. Implement the versioned codec with bounded structural validation and safe unsupported-dtype errors; retain old encodings and reject tampered payloads atomically.
- [x] Add `test_time_research_pack_preserves_submillisecond_splits`: 100 timestamps one microsecond apart, time validation, real comparison, export and independent replay. Assert 100 distinct timestamps, exact splits, recorded scores reproduced; add nanosecond/timezone/tied-timestamp cases. Observe old export/replay failure.
- [x] Switch new research exports to the codec with manifest schema dispatch and update replay-source lists for source and frozen execution. Persist original source provenance without treating current source contents as proof of the embedded frame's identity.
- [x] Add a checked-in small legacy schema-1 pack fixture/fixture builder and legacy project fixtures, then verify compatibility or explicit previously-lossy-data diagnostics. Do not fabricate recovery of data already lost in an old export.
- [x] Run targeted `python -m pytest -q tests/test_project_fidelity.py tests/test_research.py tests/test_audit_fixes.py tests/test_student_upgrade.py tests/test_arff_io.py`, then the full suite, using the explicit virtual environment interpreter.
- [x] Native acceptance: unsigned-boundary CSV and nominal ARFF project save/reopen/export; research timestamp pack export and replay in a separate process. Model training/holdout metrics must remain unchanged on existing fixtures. Validate generated replay imports and missing-source behavior.
- [x] Update README/status/format documentation with new and backward-compatible schemas, exact tested types and unsupported-type behavior; review completed; changes remain local with no commit/push claim. Installer/clean-machine acceptance remains open.

## Task 3: One recorded-holdout validator for diagnostic evidence

**Files:**
- Create `src/mlstudio/holdout_validation.py` as a small shared validator extracted from the existing implementation.
- Modify `src/mlstudio/model_diagnostics.py` and `evaluation.py` to consume it without altering fit/prediction/metric behavior for valid inputs.
- Modify replay-source lists in `research_export.py` and `packaging/mlstudio.spec`: include `holdout_validation.py` and `data.py`, which supplies `schema_fingerprint`. Preserve the Task 2 codec source and verify the complete import dependency closure; do not rely on the developer checkout making missing package modules importable.
- Extend `tests/test_model_diagnostics.py`, `test_student_upgrade.py`; create `tests/test_holdout_validation.py`.

**Interfaces:**
- Produce `validate_recorded_holdout(result: TrainingResult, frame: pd.DataFrame) -> tuple[list[str], str, list[int]]`, returning validated features, target and positional test rows; performs no fitting or inference.
- Validate result/frame shape, ordered recorded columns, `schema_fingerprint`, value/index fingerprint, nonempty unique integer positions within bounds matching `test_rows`, valid distinct features/target and nonmissing recorded outcomes. Keep `model_diagnostics` prediction/plot logic separate.
- Legacy models lacking the recorded evidence schema may still predict through the existing inference path; explaining them returns an actionable retrain/restore error. Never guess a replacement holdout.

- [ ] Add `test_explanation_rejects_renamed_same_hash_columns`: train on x/z/label, swap x/z names and assert ValueError before any permutation prediction. Observe current failure to reject.
- [ ] Add bool→integer and nominal-domain change, duplicate/noninteger/out-of-range/incorrect-count holdout positions, missing metadata/outcome, altered values/index, duplicate names and valid unchanged-frame cases. Parameterize both diagnostic and permutation-importance callers.
- [ ] Run new tests before changing behavior; retain tests proving existing ROC/PR/calibration/regression evidence does not refit.
- [ ] Extract the currently stronger validation into `validate_recorded_holdout`; use it from both callers before their deterministic sampling or inference. Remove duplicate fingerprint logic; avoid broad training/renderer refactors.
- [ ] Verify valid permutation importance results and diagnostic metrics/rows remain equal to baseline for fixed fixtures; tampered evidence never calls fit or substitutes other data.
- [ ] Run targeted `python -m pytest -q tests/test_holdout_validation.py tests/test_model_diagnostics.py tests/test_student_upgrade.py tests/test_research.py tests/test_audit_fixes.py`, then the full suite, using `.venv\Scripts\python.exe`.
- [ ] Native acceptance: valid trained/saved model explanation and charts work; restoring a wrong schema shows a clear error and no stale explanation; model-only prediction still works without the training frame. Run extracted research replay in a separate process outside the checkout with inherited PYTHONPATH removed, and assert every imported `mlstudio` module resolves inside the extracted package. This verifies the full helper dependency closure rather than allowing developer source to hide missing bundled files.
- [ ] Synchronize README/status and review before an authorized task commit. Explicitly leave model-manifest hardening, calibrated uncertainty, SHAP and broader P0 work open.

## After these three tasks

Return to the audit's P0-B list before implementing new search/AutoML/node catalogs. Keep phase gates small: tests first, targeted suite, full suite, applicable UI acceptance, documentation, review. No installer, competitor parity, all-functionality correctness or universal model-quality claim follows from these tasks. Any broad rewrite requires a separate concrete technical justification and the user's approval.

## 29 October implementation record for Task 3

Implemented as `holdout.py::validate_holdout` (names differ from the proposal) and consumed by diagnostics and permutation importance. Added fitted feature-order agreement and revision-safe explanation invalidation. `test_holdout_contract.py` covers 19 shared/Qt cases; existing diagnostics/student tests retain no-refit and actual metric evidence. Native EXE acceptance includes valid explanation, incompatible evidence rejection, model save/open, model-only prediction and isolated research replay. Replay source lists include the new helper and data module. Additional authorized P0 fixes are recorded in `docs/RELEASE_READINESS.md`, including model-manifest hardening that was deliberately excluded from the initial three-task plan. Earlier unchecked items describe the proposed matrix, not a claim that every adversarial mutation has been tested.
