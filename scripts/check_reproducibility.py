"""Audit without changing original data/models/results; optional real GPU replay and training smoke."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROBS = [f'prob_{x}' for x in ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true', help='Replay all saved-model val/test predictions and run each original trainer on a tiny stratified dataset (1 epoch).')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproducibility')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is not empty; use a new --output to preserve previous evidence.')
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    environment = {'python': sys.version, 'platform': platform.platform(), 'packages': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'datasets', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'soundfile', 'accelerate']}}
    try:
        environment['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        environment['nvidia_smi'] = 'Unavailable'
    source_paths = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'configs').glob('*.json')) + [Path(__file__).resolve(), ROOT / 'requirements.txt']
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    env = dict(os.environ, MPLBACKEND='Agg', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    def run(name, script, cwd, extra=()):
        start = time.monotonic()
        with (out / f'{name}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(ROOT / script), *map(str, extra)], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {'name': name, 'status': 'pass' if result.returncode == 0 else 'fail', 'seconds': round(time.monotonic() - start, 2), 'log': f'{name}.log'}
        checks.append(record)
        print(record, flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed: see {out / record["log"]}')
    def compare_csv(name, actual, reference):
        a, b = pd.read_csv(actual), pd.read_csv(reference)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=1e-12)
        checks.append({'name': name, 'status': 'pass', 'rows': len(a)})
    def compare_predictions(name, actual, reference):
        a, b = pd.read_csv(actual), pd.read_csv(reference)
        if a.id.tolist() != b.id.tolist():
            raise ValueError(f'{name}: IDs differ')
        max_error = float(np.abs(a[PROBS].to_numpy() - b[PROBS].to_numpy()).max())
        mismatch_mask = a.predicted_label != b.predicted_label
        mismatches = int(mismatch_mask.sum())
        if mismatches:
            pd.DataFrame({'id': a.loc[mismatch_mask, 'id'],
                          'original_prediction': b.loc[mismatch_mask, 'predicted_label'],
                          'replay_prediction': a.loc[mismatch_mask, 'predicted_label'],
                          'max_probability_abs_difference': np.abs(a[PROBS].to_numpy() - b[PROBS].to_numpy()).max(axis=1)[mismatch_mask]}).to_csv(out / f'{name}_mismatches.csv', index=False)
        passed = mismatches == 0 and max_error <= 1e-5
        checks.append({'name': name, 'status': 'pass' if passed else 'different', 'rows': len(a), 'prediction_mismatches': mismatches, 'max_probability_abs_difference': max_error, 'tolerance': 1e-5})
    failure = None
    try:
        with tempfile.TemporaryDirectory(prefix='moodjournal-audit-') as tmp:
            work = Path(tmp)
            (work / 'data').mkdir()
            (work / 'data/raw').symlink_to(ROOT / 'data/raw', target_is_directory=True)
            shutil.copytree(ROOT / 'configs', work / 'configs')
            for script in ['build_manifest', 'validate_manifest', 'link_audio_manifest', 'validate_audio', 'mark_audio_valid']:
                run(script, f'src/utils/{script}.py', work)
            for name in ['manifest.csv', 'manifest_multimodal.csv']:
                compare_csv(f'exact_{name}', work / 'data/processed' / name, ROOT / 'data/processed' / name)
            compare_csv('audio_metadata', work / 'results/audio_validation.csv', ROOT / 'results/audio_validation.csv')
            for slug in ['text_macbert_base', 'speech_wavlm_base_plus']:
                (work / 'results' / slug).mkdir(parents=True)
                for name in ['val_predictions.csv', 'predictions.csv']:
                    shutil.copy2(ROOT / 'results' / slug / name, work / 'results' / slug / name)
            for script in ['search_weighted_fusion', 'evaluate_weighted_fusion', 'train_learned_fusion', 'train_learned_fusion_balanced']:
                run(script, f'src/fusion/{script}.py', work)
            for name in ['best_alpha.csv', 'alpha_search.csv']:
                compare_csv(name, work / 'results/weighted_fusion' / name, ROOT / 'results/weighted_fusion' / name)
            for slug in ['weighted_fusion', 'learned_fusion_logreg', 'learned_fusion_logreg_balanced']:
                compare_csv(f'{slug}_predictions', work / 'results' / slug / 'test_predictions.csv', ROOT / 'results' / slug / 'test_predictions.csv')
                a = json.loads((work / 'results' / slug / 'test_metrics.json').read_text())
                b = json.loads((ROOT / 'results' / slug / 'test_metrics.json').read_text())
                if a != b:
                    raise ValueError(f'{slug}: metrics JSON differs')
                checks.append({'name': f'{slug}_metrics', 'status': 'pass'})
            run('summary', 'src/analysis/summarize_results.py', ROOT, ['--output', work / 'summary'])
            if args.models:
                (work / 'models').symlink_to(ROOT / 'models', target_is_directory=True)
                run('replay_validation', 'src/fusion/generate_val_predictions.py', work)
                run('replay_test', 'src/analysis/replay_test_predictions.py', ROOT, ['--output', work / 'replay_test'])
                for slug in ['text_macbert_base', 'speech_wavlm_base_plus']:
                    compare_predictions(f'{slug}_validation_replay', work / 'results' / slug / 'val_predictions.csv', ROOT / 'results' / slug / 'val_predictions.csv')
                    compare_predictions(f'{slug}_test_replay', work / 'replay_test' / slug / 'predictions.csv', ROOT / 'results' / slug / 'predictions.csv')
                # The original trainers run unchanged with tiny data/one epoch in a separate directory.
                # Training uses the original pretrained identifier (offline cache), not fine-tuned weights.
                smoke = work / 'smoke'
                (smoke / 'data/processed').mkdir(parents=True)
                (smoke / 'data/raw').symlink_to(ROOT / 'data/raw', target_is_directory=True)
                (smoke / 'configs').mkdir()
                full = pd.read_csv(work / 'data/processed/manifest_multimodal.csv')
                tiny = pd.concat([group.groupby('label', sort=False).head(2 if split == 'train' else 1) for split, group in full.groupby('split', sort=False)])
                tiny.to_csv(smoke / 'data/processed/manifest_multimodal.csv', index=False)
                for modality, filename in [('text', 'text_macbert'), ('speech', 'speech_wavlm')]:
                    config = json.loads((ROOT / f'configs/{filename}.json').read_text())
                    config['num_train_epochs'] = 1
                    (smoke / f'configs/{filename}.json').write_text(json.dumps(config))
                    run(f'{modality}_train_smoke', f'src/{modality}/train_{modality}.py', smoke)
                checks.append({'name': 'training_smoke_scope', 'status': 'pass', 'train': 14, 'val': 7, 'test': 7, 'epochs': 1, 'initialization': 'original pretrained models from local cache'})
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        checks.append({'name': 'audit_exception', 'status': 'fail', 'detail': failure})
    payload = {'environment': environment, 'source_sha256': source_hashes, 'checks': checks, 'full_retraining': 'NOT RUN: smoke is one epoch on 14 training utterances, not five epochs on the full dataset.', 'fresh_environment_install': 'NOT RUN: audit uses existing installed environment.', 'model_checks_requested': args.models, 'failure': failure}
    (out / 'audit.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Audit written to {out / "audit.json"}', flush=True)
    if failure or any(c['status'] in ['fail', 'different'] for c in checks):
        sys.exit(1)


if __name__ == '__main__':
    main()
