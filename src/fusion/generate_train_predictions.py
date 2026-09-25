import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


MANIFEST_PATH = Path(
    "data/processed/manifest_multimodal.csv"
)

TEXT_CONFIG = Path(
    "configs/text_macbert.json"
)

SPEECH_CONFIG = Path(
    "configs/speech_wavlm.json"
)

TEXT_MODEL_DIR = Path(
    "models/text_macbert_base/best_model"
)

SPEECH_MODEL_DIR = Path(
    "models/speech_wavlm_base_plus/best_model"
)

TEXT_OUTPUT = Path(
    "results/text_macbert_base/train_predictions.csv"
)

SPEECH_OUTPUT = Path(
    "results/speech_wavlm_base_plus/train_predictions.csv"
)


def load_configs():

    with open(
        TEXT_CONFIG,
        "r",
        encoding="utf-8",
    ) as f:
        text_config = json.load(f)

    with open(
        SPEECH_CONFIG,
        "r",
        encoding="utf-8",
    ) as f:
        speech_config = json.load(f)

    if text_config["labels"] != speech_config["labels"]:
        raise ValueError(
            "Text and Speech label order is different."
        )

    return text_config, speech_config


def save_predictions(
    df,
    probabilities,
    labels,
    output_path,
):

    pred_ids = np.argmax(
        probabilities,
        axis=1,
    )

    result = df[
        [
            "id",
            "text",
            "label",
            "split",
        ]
    ].copy()

    result["predicted_label"] = [
        labels[i]
        for i in pred_ids
    ]

    for i, label in enumerate(labels):
        result[f"prob_{label}"] = (
            probabilities[:, i]
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Saved:",
        output_path,
        "Rows:",
        len(result),
    )


def generate_text_predictions(
    train_df,
    config,
    device,
):

    print("\n=== TEXT VALIDATION PREDICTIONS ===")

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_DIR
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            TEXT_MODEL_DIR
        )
    )

    model.to(device)
    model.eval()

    all_probs = []

    batch_size = 64

    with torch.inference_mode():

        for start in range(
            0,
            len(train_df),
            batch_size,
        ):

            batch_df = train_df.iloc[
                start:start + batch_size
            ]

            inputs = tokenizer(
                batch_df["text"].astype(str).tolist(),
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt",
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            logits = model(
                **inputs
            ).logits

            probs = torch.softmax(
                logits.float(),
                dim=-1,
            )

            all_probs.append(
                probs.cpu().numpy()
            )

    probabilities = np.concatenate(
        all_probs,
        axis=0,
    )

    print(
        "Shape:",
        probabilities.shape,
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return probabilities


def generate_speech_predictions(
    train_df,
    config,
    device,
):

    print("\n=== SPEECH VALIDATION PREDICTIONS ===")

    feature_extractor = (
        AutoFeatureExtractor
        .from_pretrained(
            SPEECH_MODEL_DIR
        )
    )

    model = (
        AutoModelForAudioClassification
        .from_pretrained(
            SPEECH_MODEL_DIR
        )
    )

    model.to(device)
    model.eval()

    all_probs = []

    batch_size = 8

    with torch.inference_mode():

        for start in range(
            0,
            len(train_df),
            batch_size,
        ):

            batch_df = train_df.iloc[
                start:start + batch_size
            ]

            audio_arrays = []

            for path in batch_df[
                "audio_path"
            ]:

                audio, sr = sf.read(
                    path,
                    dtype="float32",
                )

                if sr != config[
                    "sampling_rate"
                ]:
                    raise ValueError(
                        f"Unexpected sample rate: {sr}"
                    )

                if len(audio) == 0:
                    raise ValueError(
                        f"Empty audio: {path}"
                    )

                audio_arrays.append(
                    audio
                )

            inputs = feature_extractor(
                audio_arrays,
                sampling_rate=config[
                    "sampling_rate"
                ],
                padding=True,
                return_tensors="pt",
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            logits = model(
                **inputs
            ).logits

            probs = torch.softmax(
                logits.float(),
                dim=-1,
            )

            all_probs.append(
                probs.cpu().numpy()
            )

            if (
                start + batch_size
            ) % 500 < batch_size:

                print(
                    f"Processed "
                    f"{min(start + batch_size, len(train_df))}"
                    f"/{len(train_df)}"
                )

    probabilities = np.concatenate(
        all_probs,
        axis=0,
    )

    print(
        "Shape:",
        probabilities.shape,
    )

    return probabilities


def main():

    text_config, speech_config = (
        load_configs()
    )

    labels = text_config["labels"]

    df = pd.read_csv(
        MANIFEST_PATH
    )

    train_df = df[
        df["split"] == "train"
    ].reset_index(drop=True)

    print("=== VALIDATION SET ===")
    print("Rows:", len(train_df))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    text_probs = generate_text_predictions(
        train_df,
        text_config,
        device,
    )

    save_predictions(
        train_df,
        text_probs,
        labels,
        TEXT_OUTPUT,
    )

    speech_probs = generate_speech_predictions(
        train_df,
        speech_config,
        device,
    )

    save_predictions(
        train_df,
        speech_probs,
        labels,
        SPEECH_OUTPUT,
    )

    print("\n=== FINAL CHECK ===")

    print(
        "Text probabilities:",
        text_probs.shape,
    )

    print(
        "Speech probabilities:",
        speech_probs.shape,
    )

    assert (
        text_probs.shape
        == speech_probs.shape
        == (len(train_df), len(labels))
    )

    print(
        "Train predictions ready!"
    )


if __name__ == "__main__":
    main()
