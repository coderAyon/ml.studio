"""Read-only plots from a fitted model's recorded holdout; never fit or tune here."""
import base64
import io

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_curve, precision_recall_fscore_support,
    roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, r2_score)

from .training import REGRESSION, TrainingResult, prediction_inputs


CAUTION = ('These are the recorded holdout rows, not a new independent test. '
           'Do not select or tune a model repeatedly against these plots; reserve new data for final evaluation.')


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _table(columns, rows):
    return {'columns': columns, 'data': [[_scalar(v) for v in row] for row in rows],
            'index': list(range(len(rows)))}


def _chart(kind, title, draw, help_text):
    figure = Figure(figsize=(6.2, 4.2), dpi=110, layout='constrained')
    axes = figure.subplots()
    axes.spines[['top', 'right']].set_visible(False)
    axes.grid(alpha=.15)
    draw(axes)
    axes.set_title(title, loc='left', fontsize=12, pad=14)
    buffer = io.BytesIO()
    figure.savefig(buffer, format='svg', bbox_inches='tight')
    return {'id': kind, 'title': title, 'help': help_text,
            'image': 'data:image/svg+xml;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')}


def _holdout(result, frame):
    from .holdout import validate_holdout
    features, target, positions = validate_holdout(result, frame)
    x = prediction_inputs(result, frame.iloc[positions][features])
    actual = frame.iloc[positions][target].to_numpy()
    predicted = np.asarray(result.pipeline.predict(x))
    if predicted.ndim != 1 or len(predicted) != len(actual):
        raise ValueError('This model returned an unsupported prediction shape.')
    return x, actual, predicted, positions


def build_diagnostics(result, frame, positive_class=None):
    """Compute true holdout evidence, validating dataset identity before inference.

    Tables contain the full recorded holdout (up to 8,000 rows). UI may display
    a bounded preview; charts never silently exclude rows from their metrics.
    """
    x, actual, predicted, positions = _holdout(result, frame)
    meta = result.metadata
    report = {'algorithm': meta['algorithm'], 'target': meta['target'],
              'holdout_rows': len(actual), 'training_rows': meta.get('training_rows'),
              'dataset_fingerprint': meta['dataset_fingerprint'], 'caution': CAUTION,
              'charts': [], 'notes': [], 'classes': [], 'positive_class': None,
              'tables': {}, 'metrics': {}}
    if meta['algorithm'] in REGRESSION:
        report['task'] = 'regression'
        actual, predicted = actual.astype(float), predicted.astype(float)
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError('Regression holdout outcomes and predictions must be finite.')
        residual = actual - predicted
        report['metrics'] = {'MAE': float(mean_absolute_error(actual, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'R2': _scalar(r2_score(actual, predicted)) if len(actual) > 1 else None,
            'Mean residual': float(residual.mean())}
        report['tables']['predictions'] = _table(
            ['Active row', 'Actual', 'Predicted', 'Residual (actual - predicted)', 'Absolute error'],
            [[p+1, a, b, e, abs(e)] for p, a, b, e in zip(positions, actual, predicted, residual)])
        # Rendering is sampled only when needed; all metrics and export rows use every holdout row.
        chosen = np.linspace(0, len(actual)-1, min(1500, len(actual))).astype(int)
        low, high = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
        def predicted_plot(ax):
            ax.scatter(actual[chosen], predicted[chosen], s=15, alpha=.6, color='#7853cf')
            ax.plot([low, high], [low, high], '--', color='#7b849b', linewidth=1)
            ax.set(xlabel='Actual answer', ylabel='Model prediction')
        def residual_plot(ax):
            ax.scatter(predicted[chosen], residual[chosen], s=15, alpha=.6, color='#268e91')
            ax.axhline(0, color='#7b849b', linestyle='--', linewidth=1)
            ax.set(xlabel='Model prediction', ylabel='Actual − predicted')
        report['charts'] = [
            _chart('predicted_actual', 'Predicted vs actual', predicted_plot,
                'Points near the dashed diagonal are closer predictions.'),
            _chart('residuals', 'Where predictions miss', residual_plot,
                'Positive residuals mean the model predicted too low. Look for patterns around zero.')]
        if len(actual) > len(chosen):
            report['notes'].append(f'Plots show {len(chosen)} evenly selected holdout records; scores and CSV use all {len(actual)}.')
        return report

    report['task'] = 'classification'
    classes = np.asarray(result.pipeline.named_steps['model'].classes_)
    if classes.ndim != 1 or len(classes) < 2:
        raise ValueError('Classification diagnostics need a fitted model with at least two classes.')
    names = [str(c) for c in classes]
    if len(names) != len(set(names)):
        raise ValueError('Class names are ambiguous after conversion to text. Review labels and train again.')
    selected = str(positive_class) if positive_class is not None else names[-1]
    if selected not in names:
        raise ValueError('Choose a positive class from this model’s class labels.')
    index = names.index(selected)
    truth = actual == classes[index]
    report.update({'classes': names, 'positive_class': selected})
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=classes, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=classes)
    report['tables']['per_class'] = _table(['Class', 'Precision', 'Recall', 'F1', 'Test records'],
        list(zip(names, precision, recall, f1, support)))
    report['tables']['confusion'] = _table(['Actual / predicted', *names],
        [[name, *row] for name, row in zip(names, matrix)])
    report['tables']['predictions'] = _table(['Active row', 'Actual', 'Predicted', 'Correct'],
        [[p+1, a, b, bool(a == b)] for p, a, b in zip(positions, actual, predicted)])
    report['metrics'] = {'Accuracy': float(np.mean(actual == predicted)),
        'Macro F1': float(np.mean(f1)), 'Positive test records': int(truth.sum()),
        'Other test records': int((~truth).sum())}
    scores = probabilities = None
    if hasattr(result.pipeline, 'predict_proba'):
        raw = np.asarray(result.pipeline.predict_proba(x), dtype=float)
        if raw.shape != (len(actual), len(classes)) or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
            raise ValueError('This model returned invalid class probabilities.')
        scores = probabilities = raw[:, index]
        report['score_kind'] = 'probability'
    elif hasattr(result.pipeline, 'decision_function'):
        raw = np.asarray(result.pipeline.decision_function(x), dtype=float)
        if raw.ndim == 1 and len(classes) == 2:
            scores = raw if index == 1 else -raw
        elif raw.shape == (len(actual), len(classes)):
            scores = raw[:, index]
        report['score_kind'] = 'decision score'
    else:
        report['score_kind'] = None
    if scores is not None and (np.asarray(scores).shape != (len(actual),) or not np.isfinite(scores).all()):
        raise ValueError('This model returned invalid decision scores.')
    if scores is None:
        report['notes'].append('This estimator exposes labels only. ROC, precision–recall and calibration require scores or probabilities.')
        return report
    if len(np.unique(truth)) < 2:
        report['notes'].append('This holdout does not contain both the selected class and other classes. ROC and precision–recall are unavailable for this selection.')
        return report
    fpr, tpr, roc_thresholds = roc_curve(truth, scores)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(truth, scores)
    auc = float(roc_auc_score(truth, scores))
    ap = float(average_precision_score(truth, scores))
    prevalence = float(truth.mean())
    report['metrics'].update({'ROC AUC': auc, 'Average precision': ap, 'Positive prevalence': prevalence})
    curves = [['ROC', x, y, threshold] for x, y, threshold in zip(fpr, tpr, roc_thresholds)]
    curves += [['Precision–recall', x, y, threshold] for x, y, threshold in
               zip(pr_recall, pr_precision, [*pr_thresholds, None])]
    report['tables']['curves'] = _table(['Curve', 'X', 'Y', 'Threshold'], curves)
    def roc_plot(ax):
        ax.plot(fpr, tpr, color='#7853cf', linewidth=2, label=f'AUC {auc:.3f}')
        ax.plot([0, 1], [0, 1], '--', color='#7b849b', linewidth=1)
        ax.set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc='lower right', frameon=False)
    def pr_plot(ax):
        ax.step(pr_recall, pr_precision, where='post', color='#268e91', linewidth=2, label=f'AP {ap:.3f}')
        ax.axhline(prevalence, color='#7b849b', linestyle='--', linewidth=1, label='Class prevalence')
        ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc='lower left', frameon=False)
    report['charts'] = [
        _chart('roc', f'ROC · {selected} vs the rest', roc_plot,
            'Shows the trade-off between finding this class and false alarms across thresholds.'),
        _chart('precision_recall', f'Precision–recall · {selected}', pr_plot,
            'Useful for rare classes. The dashed reference is this class’s share of the holdout.')]
    if probabilities is None:
        report['notes'].append('Calibration is unavailable because this model returns decision scores, not probabilities. Scores were not converted to pretend probabilities.')
    else:
        bins = []
        for b in range(10):
            low, high = b/10, (b+1)/10
            mask = (probabilities >= low) & ((probabilities <= high) if b == 9 else (probabilities < high))
            if mask.any():
                bins.append([low, high, int(mask.sum()), float(probabilities[mask].mean()), float(truth[mask].mean())])
        report['tables']['calibration'] = _table(
            ['Bin from', 'Bin to', 'Test records', 'Mean predicted probability', 'Observed positive fraction'], bins)
        brier = float(brier_score_loss(truth, probabilities))
        report['metrics']['Brier score'] = brier
        def calibration_plot(ax):
            ax.plot([row[3] for row in bins], [row[4] for row in bins], 'o-', color='#bd7d28', linewidth=2)
            ax.plot([0, 1], [0, 1], '--', color='#7b849b', linewidth=1)
            ax.set(xlabel='Mean predicted probability', ylabel='Observed positive fraction', xlim=(0, 1), ylim=(0, 1.02))
        report['charts'].append(_chart('calibration', f'Probability reliability · {selected}', calibration_plot,
            'Ten equal-width probability bins; empty bins are omitted. Near the diagonal means closer agreement. Small bins are uncertain.'))
    return report


def diagnostics_rows(report, kind):
    """Export recorded plot/table values; never recompute or refit on export."""
    if not isinstance(report, dict) or kind not in report.get('tables', {}):
        raise ValueError('Build diagnostics and choose an available table before exporting.')
    output = report['tables'][kind]
    return pd.DataFrame(output['data'], columns=output['columns'])
