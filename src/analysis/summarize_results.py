"""Rebuild publication tables/figures from saved predictions; fail on inconsistent evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
METHODS = {
    'text_macbert_base': 'MacBERT (text)',
    'speech_wavlm_base_plus': 'WavLM (speech)',
    'weighted_fusion': 'Weighted fusion',
    'learned_fusion_logreg': 'Learned fusion (LR)',
    'learned_fusion_logreg_balanced': 'Balanced learned fusion (LR)',
}
PROBS = [f'prob_{label}' for label in LABELS]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_predictions(path, expected, label_column='true_label'):
    df = pd.read_csv(path)
    require(not df.id.isna().any() and df.id.is_unique, f'{path}: missing/duplicate IDs')
    require(set(df.id) == set(expected.id), f'{path}: split coverage mismatch')
    df = df.set_index('id').loc[expected.id].reset_index()
    require(df[label_column].tolist() == expected.label.tolist(), f'{path}: labels mismatch')
    require(df.predicted_label.isin(LABELS).all(), f'{path}: unknown prediction')
    p = df[PROBS].to_numpy()
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), f'{path}: invalid probabilities')
    require(np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0), f'{path}: probabilities do not sum to 1')
    require(np.array_equal(np.array(LABELS)[p.argmax(axis=1)], df.predicted_label), f'{path}: argmax mismatch')
    return df


def markdown(frame, percent=(), bold_best=False):
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for _, row in frame.iterrows():
        values = []
        for col, value in row.items():
            if col in percent:
                formatted = f'{value * 100:.2f}'
                if bold_best and np.isclose(value, frame[col].max(), rtol=0, atol=1e-12):
                    formatted = f'**{formatted}**'
            else:
                formatted = str(value)
            values.append(formatted)
        rows.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(rows)


def draw_matrix(ax, cm, title, normalized, vmax):
    values = cm / cm.sum(axis=1, keepdims=True) if normalized else cm
    im = ax.imshow(values, cmap='Blues', vmin=0, vmax=vmax)
    ax.set(title=title, xlabel='Predicted emotion', ylabel='True emotion')
    ax.set_xticks(range(7), LABELS, rotation=45, ha='right')
    ax.set_yticks(range(7), [f'{label} (n={n})' for label, n in zip(LABELS, cm.sum(axis=1))])
    for i in range(7):
        for j in range(7):
            text = f'{values[i, j] * 100:.1f}%' if normalized else str(cm[i, j])
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if values[i, j] > vmax * 0.55 else '#172433')
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / 'results/summary'
    manifest_path = root / 'data/processed/manifest_multimodal.csv'
    manifest = pd.read_csv(manifest_path)
    require(manifest.id.is_unique, 'Manifest IDs must be unique')
    require(manifest.label.isin(LABELS).all(), 'Unknown manifest labels')
    require(set(manifest.split) == {'train', 'val', 'test'}, 'Unexpected splits')
    require(manifest.groupby('movie').split.nunique().max() == 1, 'Movie leakage')
    require(manifest.groupby(['movie', 'scene_id']).split.nunique().max() == 1, 'Scene leakage')
    expected = manifest.loc[manifest.split == 'test'].reset_index(drop=True)
    inputs = [manifest_path, Path(__file__).resolve()]
    overall, per_class, matrices = [], [], {}
    for slug, name in METHODS.items():
        base = slug in ('text_macbert_base', 'speech_wavlm_base_plus')
        directory = root / 'results' / slug
        prediction_path = directory / ('predictions.csv' if base else 'test_predictions.csv')
        metrics_path = directory / ('metrics.json' if base else 'test_metrics.json')
        df = checked_predictions(prediction_path, expected)
        saved = json.loads(metrics_path.read_text())
        report = classification_report(df.true_label, df.predicted_label, labels=LABELS, output_dict=True, zero_division=0)
        cm = confusion_matrix(df.true_label, df.predicted_label, labels=LABELS)
        matrices[slug] = cm
        # Independent arithmetic cross-check from the confusion matrix.
        tp, support, predicted = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
        recall = tp / support
        precision = np.divide(tp, predicted, out=np.zeros(7), where=predicted != 0)
        f1 = np.divide(2 * tp, support + predicted, out=np.zeros(7), where=support + predicted != 0)
        row = {'Method': name, 'Accuracy': accuracy_score(df.true_label, df.predicted_label),
               'Macro-F1': float(f1.mean()), 'Macro-Precision': float(precision.mean()),
               'UAR': float(recall.mean()), 'Weighted-F1': report['weighted avg']['f1-score'], 'N': len(df)}
        for column, key in [('Accuracy', 'test_accuracy'), ('Macro-F1', 'test_macro_f1'), ('Macro-Precision', 'test_macro_precision'), ('UAR', 'test_macro_recall' if slug == 'text_macbert_base' else 'test_uar')]:
            require(abs(row[column] - saved[key]) < 1e-12, f'{slug}: {key} differs from saved JSON')
        for i, label in enumerate(LABELS):
            for key, computed in [('precision', precision[i]), ('recall', recall[i]), ('f1-score', f1[i]), ('support', support[i])]:
                require(abs(report[label][key] - computed) < 1e-12, f'{slug}: independent metric check failed')
                require(abs(saved['classification_report'][label][key] - computed) < 1e-12, f'{slug}: saved class metric mismatch')
            per_class.append({'Method': name, 'Emotion': label, 'Precision': precision[i], 'Recall': recall[i], 'F1': f1[i], 'Support': int(support[i]), 'TP': int(tp[i]), 'Predicted': int(predicted[i]), 'FP': int(predicted[i] - tp[i]), 'FN': int(support[i] - tp[i])})
        overall.append(row)
        inputs.extend([prediction_path, metrics_path])
    # Validate both base models' validation populations as the fusion training source.
    val = manifest.loc[manifest.split == 'val'].reset_index(drop=True)
    for slug in list(METHODS)[:2]:
        path = root / 'results' / slug / 'val_predictions.csv'
        checked_predictions(path, val, 'label')
        inputs.append(path)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'confusion_matrices'
    figures.mkdir(exist_ok=True)
    overall = pd.DataFrame(overall)
    per_class = pd.DataFrame(per_class)
    distribution = pd.crosstab(manifest.label, manifest.split).reindex(index=LABELS, columns=['train', 'val', 'test'])
    distribution.columns.name = None
    distribution.index.name = 'Emotion'
    counts = distribution.copy()
    for split in ['train', 'val', 'test']:
        distribution[f'{split}_share'] = counts[split] / counts[split].sum()
    distribution['balanced_LR_weight'] = len(val) / (7 * counts.val)
    overall.to_csv(out / 'overall_results.csv', index=False)
    per_class.to_csv(out / 'per_emotion_metrics.csv', index=False)
    distribution.to_csv(out / 'class_distribution.csv')
    for metric in ['Recall', 'F1']:
        pivot = per_class.pivot(index='Emotion', columns='Method', values=metric).reindex(index=LABELS, columns=METHODS.values())
        pivot.insert(0, 'Test support', counts.test)
        pivot.to_csv(out / f'per_emotion_{metric.lower()}.csv')
    rates = ['Accuracy', 'Macro-F1', 'Macro-Precision', 'UAR', 'Weighted-F1']
    sections = ['# Experimental results', 'Generated by `python src/analysis/summarize_results.py`. All tables use the same held-out test samples, fixed seven-class order, and `zero_division=0`. Rates below are percentages; CSV rates are fractions. Bold marks the column maximum among the five methods. One seed (42); no significance claim or confidence interval.', '## Overall Results Table', markdown(overall, rates, True), 'UAR = macro-average recall. Macro-F1 averages seven per-class F1 values; it is not the harmonic mean of macro-precision and UAR. Weighted-F1 is support-weighted and supplementary.', '## Class distribution', markdown(counts.reset_index()), '## Class shares (%) and fusion weights', markdown(distribution.reset_index().drop(columns=['train', 'val', 'test']).round({'balanced_LR_weight': 4}), ['train_share', 'val_share', 'test_share'])]
    for metric in ['Recall', 'F1']:
        pivot = per_class.pivot(index='Emotion', columns='Method', values=metric).reindex(index=LABELS, columns=METHODS.values()).reset_index()
        pivot.insert(1, 'Test support', counts.test.to_numpy())
        sections.extend([f'## Per-emotion {metric} (%)', markdown(pivot, list(METHODS.values()))])
    sections.extend(['## Confusion matrices', 'Rows = true emotion; columns = predicted emotion. Each row includes its test support. Count panels share one scale across methods; normalized panels share 0–100%. Each normalized diagonal is the corresponding class recall. Zero counts are shown explicitly.', '![Row-normalized confusion matrices](confusion_matrices/all_normalized.png)', '![Count confusion matrices](confusion_matrices/all_counts.png)', 'Individual PNG/PDF figures and numerical count/row-normalized CSV matrices are in `confusion_matrices/`. Source hashes and validation results are in `provenance.json`.'])
    (out / 'RESULTS.md').write_text('\n\n'.join(sections) + '\n')
    # A self-contained LaTeX table requiring booktabs; retain full precision in CSV.
    latex = ['% Requires \\usepackage{booktabs}', '\\begin{table}[t]', '\\centering', '\\caption{M3ED test results (\\%). All methods use the same 4,198 utterances; one seed (42). UAR is macro recall. Bold indicates the best score among these five methods.}', '\\label{tab:overall}', '\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Accuracy & Macro-F1 & Macro-P & UAR \\\\', '\\midrule']
    for _, row in overall.iterrows():
        values = []
        for key in rates[:4]:
            value = f'{row[key] * 100:.2f}'
            if row[key] == overall[key].max():
                value = '\\textbf{' + value + '}'
            values.append(value)
        latex.append(row.Method + ' & ' + ' & '.join(values) + ' \\\\')
    latex.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}'])
    (out / 'overall_results.tex').write_text('\n'.join(latex) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12, 'figure.facecolor': 'white'})
    for normalized in [False, True]:
        kind = 'normalized' if normalized else 'counts'
        vmax = 1 if normalized else max(int(cm.max()) for cm in matrices.values())
        fig, axes = plt.subplots(2, 3, figsize=(23, 14), layout='constrained')
        for ax, (slug, cm) in zip(axes.flat, matrices.items()):
            im = draw_matrix(ax, cm, METHODS[slug], normalized, vmax)
            individual, single_ax = plt.subplots(figsize=(9, 8), layout='constrained')
            single_im = draw_matrix(single_ax, cm, METHODS[slug], normalized, vmax)
            cb = individual.colorbar(single_im, ax=single_ax, shrink=0.8)
            cb.set_label('Within true class (%)' if normalized else 'Utterances')
            if normalized:
                cb.ax.yaxis.set_major_formatter(PercentFormatter(1))
            for ext in ['png', 'pdf']:
                individual.savefig(figures / f'{slug}_{kind}.{ext}', dpi=180)
            plt.close(individual)
            values = cm / cm.sum(axis=1, keepdims=True) if normalized else cm
            pd.DataFrame(values, index=pd.Index(LABELS, name='True / Predicted'), columns=LABELS).to_csv(figures / f'{slug}_{kind}.csv')
        axes.flat[-1].axis('off')
        axes.flat[-1].text(0.05, 0.7, 'M3ED paired test set\nn = 4,198; seed = 42\n\nRows: true emotion\nColumns: predicted emotion\n\nSame color scale across all methods\n' + ('Cells: row-normalized percentages\nDiagonal: recall' if normalized else 'Cells: utterance counts'), va='top', fontsize=16)
        cb = fig.colorbar(im, ax=list(axes.flat[:5]), shrink=0.7, pad=0.025)
        cb.set_label('Within true class (%)' if normalized else 'Utterances')
        if normalized:
            cb.ax.yaxis.set_major_formatter(PercentFormatter(1))
        for ext in ['png', 'pdf']:
            fig.savefig(figures / f'all_{kind}.{ext}', dpi=160)
        plt.close(fig)
    def relative(p):
        return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
    provenance = {'source_of_truth': 'Saved per-utterance predictions, joined to paired manifest by unique ID', 'label_order': LABELS, 'test_samples': len(expected), 'checks': ['exact test/validation ID coverage and labels', 'unique IDs, no movie/scene split overlap', 'finite normalized probabilities and argmax', 'all reported metrics match saved JSON within 1e-12', 'independent confusion-matrix arithmetic matches sklearn'], 'sha256': {relative(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(overall.to_string(index=False))
    print(f'Validated and wrote {out}')


if __name__ == '__main__':
    main()
