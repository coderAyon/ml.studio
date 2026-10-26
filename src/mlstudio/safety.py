"""Evidence-based experiment design warnings; no automatic leakage verdicts."""
import pandas as pd


def safety_advice(frame, config):
    findings = []
    def add(code, title, columns, evidence, action, severity="Medium", status="Potential issue"):
        findings.append(dict(code=code,title=title,columns=columns,evidence=evidence,action=action,severity=severity,status=status))
    target = config.target
    if target not in frame:
        add("target_missing","Choose your prediction target",[],"No valid outcome is selected.","Select the column you intend to predict.","High","Configuration required")
        return findings
    y = frame[target]
    if y.isna().any():
        add("target_null","Target labels are missing",[target],f"{int(y.isna().sum())} missing labels.","Review these rows; do not invent target labels.","High","Confirmed")
    if len(frame)<100:
        add("small_sample","Limited validation data",[],f"Only {len(frame)} rows before final-test reservation.","Interpret variation cautiously; collect more examples where possible.")
    if config.task=="classification":
        counts=y.value_counts()
        if len(counts)<2 or (not counts.empty and counts.min()<8):
            add("rare_class","Some classes have too few examples",[target],str(counts.to_dict()),"Use fewer folds or obtain more labeled examples; do not silently switch to unstratified splitting.","High")
        if len(counts) and counts.max()/len(frame)>.8:
            add("imbalance","Accuracy alone may mislead",[target],f"Largest class: {counts.max()/len(frame):.1%}.","Inspect macro F1, balanced accuracy, per-class recall and a majority-class baseline.")
        if y.nunique()>20:
            add("many_classes","Review classification versus regression",[target],f"{y.nunique()} distinct target values.","If the outcome is an amount, choose regression. Do not infer research intent from numeric type.")
    elif not pd.api.types.is_numeric_dtype(y):
        add("nonnumeric_target","Regression requires a numeric target",[target],str(y.dtype),"Correct the target type or choose classification.","High","Confirmed")
    duplicate_count=int(frame.duplicated().sum())
    if duplicate_count:
        add("duplicates","Repeated records may cross random splits",[],f"{duplicate_count} exact duplicate rows.","Inspect whether these are duplicated records or legitimate repeated measurements; choose an appropriate group identifier if needed.")
    for c in frame:
        if c==target:continue
        s=frame[c]; name=str(c).lower()
        if s.astype(str).equals(y.astype(str)):
            add("target_copy","Feature repeats the target",[c],"Every value matches the selected outcome.","Investigate provenance and exclude it if it would not be available at prediction time.","High")
        elif pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y) and s.nunique()>1 and y.nunique()>1:
            corr=s.replace([float('inf'),float('-inf')],float('nan')).corr(y.replace([float('inf'),float('-inf')],float('nan')))
            if pd.notna(corr) and abs(corr)>.98:
                add("target_correlation","Very strong feature–target association",[c],f"Absolute Pearson correlation {abs(corr):.4f}.","Review availability and provenance; association alone does not prove leakage.")
        if any(token in name for token in ("patient","subject","customer","person","group","student_id")) and 1<s.nunique()<len(frame):
            add("possible_group","Possible repeated-subject identifier",[c],f"{s.nunique()} distinct values across {len(frame)} rows.","If records belong to the same entity, choose group-wise validation and this column. It will be excluded from model features.")
        if any(token in name for token in ("date","time","timestamp")):
            add("possible_time","Possible temporal ordering",[c],f"Column name '{c}' suggests time; interpretation requires review.","For future prediction choose chronological validation and a valid date/time column. The split does not infer the deployment scenario.")
        if any(token in name for token in ("future","post_outcome","after_event","result_code")):
            add("future_feature","Review feature availability",[c],"Column name suggests information that may be known after the outcome.","Confirm this value exists when predictions are made; exclude it otherwise.")
    findings.append(dict(code="protocol",title="Selected validation protocol",columns=[v for v in [config.group,config.time_column] if v],evidence=config.protocol,action="The final 20% is reserved before development CV. Group/time columns are excluded from model features. No safety scan can certify absence of leakage.",severity="Informational",status="Configuration"))
    return findings
