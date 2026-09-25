"""Compare a fresh full run against the original predictions and result tables."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference, run = args.reference.resolve(), args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    files = ['data/processed/manifest.csv', 'data/processed/manifest_multimodal.csv', 'results/audio_validation.csv', 'results/weighted_fusion/alpha_search.csv', 'results/weighted_fusion/best_alpha.csv', 'results/summary/overall_results.csv', 'results/summary/per_emotion_metrics.csv']
    for slug in ['text_macbert_base', 'speech_wavlm_base_plus']:
        files += [f'results/{slug}/predictions.csv', f'results/{slug}/val_predictions.csv']
    for slug in ['weighted_fusion', 'learned_fusion_logreg', 'learned_fusion_logreg_balanced']:
        files += [f'results/{slug}/test_predictions.csv']
    for name in files:
        a, b = pd.read_csv(reference / name), pd.read_csv(run / name)
        entry = {'file': name, 'reference_sha256': sha256(reference / name), 'run_sha256': sha256(run / name)}
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            entry['status'] = 'exact'
        except AssertionError as exc:
            entry['status'] = 'different'
            entry['detail'] = str(exc)[:500]
        if 'predicted_label' in a and a.id.tolist() == b.id.tolist():
            entry['label_mismatches'] = int((a.predicted_label != b.predicted_label).sum())
            cols = [c for c in a if c.startswith('prob_')]
            entry['max_probability_abs_difference'] = float(np.abs(a[cols].to_numpy() - b[cols].to_numpy()).max())
        entry['rows'] = len(a)
        checks.append(entry)
    weights = []
    for slug in ['text_macbert_base', 'speech_wavlm_base_plus']:
        name = f'models/{slug}/best_model/model.safetensors'
        a, b = sha256(reference / name), sha256(run / name)
        weights.append({'file': name, 'reference_sha256': a, 'run_sha256': b, 'status': 'exact' if a == b else 'different'})
    source_paths = sorted((run / 'src').rglob('*.py')) + sorted((run / 'configs').glob('*.json'))
    payload = {'reference': str(reference), 'run': str(run), 'scope': 'Full raw-data processing, both five-epoch baseline trainings, validation inference, all three fusion methods, and summary generation', 'comparison': 'Exact parsed CSV values (not just rounded metrics); weight file SHA-256', 'files': checks, 'best_model_weights': weights, 'run_source_sha256': {str(p.relative_to(run)): sha256(p) for p in source_paths}}
    (output / 'full_run_comparison.json').write_text(json.dumps(payload, indent=2) + '\n')
    original = pd.read_csv(reference / 'results/summary/overall_results.csv')
    fresh = pd.read_csv(run / 'results/summary/overall_results.csv')
    comparison = original.merge(fresh, on='Method', suffixes=('_original', '_rerun'), validate='one_to_one')
    for metric in ['Accuracy', 'Macro-F1', 'Macro-Precision', 'UAR']:
        comparison[f'{metric}_delta_pp'] = 100 * (comparison[f'{metric}_rerun'] - comparison[f'{metric}_original'])
    comparison.to_csv(output / 'full_run_metrics_comparison.csv', index=False)
    print(comparison[['Method'] + [f'{x}_delta_pp' for x in ['Accuracy', 'Macro-F1', 'Macro-Precision', 'UAR']]].to_string(index=False))
    print(f'Wrote {output / "full_run_comparison.json"}')
    if any(c['status'] != 'exact' for c in checks + weights):
        sys.exit(1)


if __name__ == '__main__':
    main()
