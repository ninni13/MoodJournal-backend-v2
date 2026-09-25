"""Replay held-out test inference from saved best models, matching training-time batch/precision."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoFeatureExtractor, AutoModelForSequenceClassification,
                          AutoModelForAudioClassification, DataCollatorWithPadding, Trainer, TrainingArguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--precision', choices=['original', 'fp32'], default='original', help='original enables BF16 on supported CUDA, matching training-time test evaluation')
    args = parser.parse_args()
    root = args.root.resolve()
    frame = pd.read_csv(root / 'data/processed/manifest_multimodal.csv')
    frame = frame.loc[frame.split == 'test'].reset_index(drop=True)
    device_bf16 = args.precision == 'original' and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    for modality, slug, config_name in [('text', 'text_macbert_base', 'text_macbert'), ('speech', 'speech_wavlm_base_plus', 'speech_wavlm')]:
        config = json.loads((root / f'configs/{config_name}.json').read_text())
        labels = config['labels']
        label2id = {label: i for i, label in enumerate(labels)}
        model_path = root / 'models' / slug / 'best_model'
        if modality == 'text':
            processor = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            data = Dataset.from_dict({'text': frame.text.tolist(), 'labels': frame.label.map(label2id).tolist()})
            data = data.map(lambda x: processor(x['text'], truncation=True, max_length=config['max_length'], padding=False), batched=True, remove_columns=['text'])
            collator = DataCollatorWithPadding(processor)
        else:
            spec = importlib.util.spec_from_file_location('speech_training', root / 'src/speech/train_speech.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            processor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForAudioClassification.from_pretrained(model_path, local_files_only=True)
            audio_frame = frame.copy()
            audio_frame['audio_path'] = audio_frame.audio_path.map(lambda p: str(root / p))
            data = module.M3EDAudioDataset(audio_frame, label2id, config['sampling_rate'])
            collator = module.AudioCollator(processor, config['sampling_rate'])
        require_mapping = {int(k): v for k, v in model.config.id2label.items()}
        if require_mapping != dict(enumerate(labels)):
            raise ValueError(f'{slug}: saved model label order differs from config')
        train_args = TrainingArguments(output_dir=str(args.output / 'trainer'), per_device_eval_batch_size=config['eval_batch_size'], bf16=device_bf16, report_to='none', remove_unused_columns=modality == 'text', dataloader_num_workers=0, seed=config['seed'], data_seed=config['seed'])
        trainer = Trainer(model=model, args=train_args, data_collator=collator, processing_class=processor)
        logits = trainer.predict(data).predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        result = frame[['id', 'movie', 'scene_id', 'speaker', 'speaker_name', 'text']].copy()
        result['true_label'] = frame.label
        result['predicted_label'] = np.array(labels)[probs.argmax(axis=1)]
        for i, label in enumerate(labels):
            result[f'prob_{label}'] = probs[:, i]
        output = args.output / slug
        output.mkdir(parents=True, exist_ok=True)
        result.to_csv(output / 'predictions.csv', index=False, encoding='utf-8-sig')
        print(f'{slug}: wrote {len(result)} predictions; bf16={device_bf16}')
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
