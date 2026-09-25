"""Run the full experiment in a new directory, preserving existing experiment artifacts."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_STEPS = ['build_manifest', 'validate_manifest', 'link_audio_manifest', 'validate_audio', 'mark_audio_valid']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path, help='New, non-existing run directory')
    parser.add_argument('--stop-after', choices=['data', 'all'], default='all')
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        parser.error('--output-dir must not already exist; historical outputs are never overwritten')
    if not (ROOT / 'data/raw/M3ED_metadata/annotation.json').is_file() or not (ROOT / 'data/raw/M3ED_audio/modality_speech').is_dir():
        parser.error('Place M3ED annotations/splits and audio in data/raw first; see README.md')
    (destination / 'data').mkdir(parents=True)
    (destination / 'data/raw').symlink_to(ROOT / 'data/raw', target_is_directory=True)
    for name in ['src', 'configs']:
        shutil.copytree(ROOT / name, destination / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / 'requirements.txt', destination / 'requirements.txt')
    (destination / 'logs').mkdir()
    env = dict(os.environ, MPLBACKEND='Agg', TOKENIZERS_PARALLELISM='false')
    steps = [f'src/utils/{name}.py' for name in DATA_STEPS]
    if args.stop_after == 'all':
        steps += ['src/text/train_text.py', 'src/speech/train_speech.py',
                  'src/fusion/generate_val_predictions.py', 'src/fusion/search_weighted_fusion.py',
                  'src/fusion/evaluate_weighted_fusion.py', 'src/fusion/train_learned_fusion.py',
                  'src/fusion/train_learned_fusion_balanced.py', 'src/analysis/summarize_results.py']
    for index, step in enumerate(steps, 1):
        log_path = destination / 'logs' / f'{index:02}_{Path(step).stem}.log'
        print(f'[{index}/{len(steps)}] {step} -> {log_path}', flush=True)
        with log_path.open('w') as log:
            result = subprocess.run([sys.executable, step], cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f'FAILED: {step}; inspect {log_path}', file=sys.stderr)
            sys.exit(result.returncode)
    print(f'Completed through {args.stop_after}: {destination}')


if __name__ == '__main__':
    main()
