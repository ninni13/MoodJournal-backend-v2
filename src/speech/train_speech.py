import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =========================================================
# Paths
# =========================================================

CONFIG_PATH = Path("configs/speech_wavlm.json")

MANIFEST_PATH = Path(
    "data/processed/manifest_multimodal.csv"
)

MODEL_DIR = Path(
    "models/speech_wavlm_base_plus"
)

RESULT_DIR = Path(
    "results/speech_wavlm_base_plus"
)


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Dataset
# =========================================================

class M3EDAudioDataset(Dataset):

    def __init__(
        self,
        dataframe,
        label2id,
        expected_sampling_rate,
    ):
        self.df = dataframe.reset_index(drop=True)

        self.label2id = label2id

        self.expected_sampling_rate = (
            expected_sampling_rate
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        audio, sample_rate = sf.read(
            row["audio_path"],
            dtype="float32",
        )

        if sample_rate != self.expected_sampling_rate:
            raise ValueError(
                f"{row['id']} has sample rate "
                f"{sample_rate}, expected "
                f"{self.expected_sampling_rate}"
            )

        if len(audio) == 0:
            raise ValueError(
                f"{row['id']} contains empty audio."
            )

        return {
            "id": row["id"],
            "audio": audio,
            "labels": self.label2id[
                row["label"]
            ],
        }


# =========================================================
# Dynamic audio padding
# =========================================================

class AudioCollator:

    def __init__(
        self,
        feature_extractor,
        sampling_rate,
    ):
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __call__(self, features):

        audio_arrays = [
            item["audio"]
            for item in features
        ]

        batch = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(
            [
                item["labels"]
                for item in features
            ],
            dtype=torch.long,
        )

        return batch


# =========================================================
# Metrics
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    # UAR = macro-average recall
    uar = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "uar": uar,
    }


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Config
    # -----------------------------------------------------

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    set_seed(
        config["seed"]
    )

    labels = config["labels"]

    label2id = {
        label: idx
        for idx, label
        in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx
        in label2id.items()
    }

    print("=== CONFIG ===")

    print(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n=== LABEL MAPPING ===")

    print(label2id)

    # -----------------------------------------------------
    # Manifest
    # -----------------------------------------------------

    df = pd.read_csv(
        MANIFEST_PATH
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print("\n=== DATASET ===")

    print(
        "Total:",
        len(df),
    )

    print(
        "Train:",
        len(train_df),
    )

    print(
        "Val:",
        len(val_df),
    )

    print(
        "Test:",
        len(test_df),
    )

    # Keep metadata for predictions.csv
    test_metadata = test_df[
        [
            "id",
            "movie",
            "scene_id",
            "speaker",
            "speaker_name",
            "text",
            "label",
            "audio_path",
        ]
    ].reset_index(drop=True)

    # -----------------------------------------------------
    # Feature extractor
    # -----------------------------------------------------

    print(
        "\nLoading feature extractor..."
    )

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            config["model_name"]
        )
    )

    print(
        "Sampling rate:",
        feature_extractor.sampling_rate,
    )

    # -----------------------------------------------------
    # Dataset objects
    # -----------------------------------------------------

    train_dataset = M3EDAudioDataset(
        train_df,
        label2id,
        config["sampling_rate"],
    )

    val_dataset = M3EDAudioDataset(
        val_df,
        label2id,
        config["sampling_rate"],
    )

    test_dataset = M3EDAudioDataset(
        test_df,
        label2id,
        config["sampling_rate"],
    )

    data_collator = AudioCollator(
        feature_extractor,
        config["sampling_rate"],
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print(
        "\nLoading WavLM..."
    )

    model = (
        AutoModelForAudioClassification
        .from_pretrained(
            config["model_name"],
            num_labels=config[
                "num_labels"
            ],
            label2id=label2id,
            id2label=id2label,
        )
    )

    # -----------------------------------------------------
    # Output folders
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Training arguments
    # -----------------------------------------------------

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    print(
        "\nBF16 enabled:",
        use_bf16,
    )

    training_args = TrainingArguments(

        output_dir=str(
            MODEL_DIR / "checkpoints"
        ),

        learning_rate=config[
            "learning_rate"
        ],

        per_device_train_batch_size=config[
            "train_batch_size"
        ],

        per_device_eval_batch_size=config[
            "eval_batch_size"
        ],

        num_train_epochs=config[
            "num_train_epochs"
        ],

        weight_decay=config[
            "weight_decay"
        ],

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        greater_is_better=True,

        save_total_limit=2,

        logging_strategy="steps",

        logging_steps=100,

        seed=config["seed"],

        data_seed=config["seed"],

        bf16=use_bf16,

        report_to="none",

        # Important for custom audio dataset/collator
        remove_unused_columns=False,

        dataloader_num_workers=4,
    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        data_collator=data_collator,

        processing_class=feature_extractor,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=
                    config[
                        "early_stopping_patience"
                    ]
            )
        ],
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print(
        "\n================================="
    )

    print(
        "Starting WavLM training"
    )

    print(
        "=================================\n"
    )

    trainer.train()

    print(
        "\nTraining finished."
    )

    print(
        "Best checkpoint:",
        trainer.state.best_model_checkpoint,
    )

    print(
        "Best validation Macro-F1:",
        trainer.state.best_metric,
    )

    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    best_model_dir = (
        MODEL_DIR / "best_model"
    )

    trainer.save_model(
        best_model_dir
    )

    feature_extractor.save_pretrained(
        best_model_dir
    )

    # -----------------------------------------------------
    # Test prediction
    # -----------------------------------------------------

    print(
        "\n================================="
    )

    print(
        "Evaluating on TEST set"
    )

    print(
        "=================================\n"
    )

    prediction_output = trainer.predict(
        test_dataset
    )

    logits = (
        prediction_output.predictions
    )

    if isinstance(logits, tuple):
        logits = logits[0]

    true_labels = (
        prediction_output.label_ids
    )

    predicted_labels = np.argmax(
        logits,
        axis=-1,
    )

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()

    # -----------------------------------------------------
    # Test metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    uar = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    print(
        "=== TEST RESULTS ==="
    )

    print(
        f"Accuracy:        "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro-F1:        "
        f"{macro_f1:.4f}"
    )

    print(
        f"Macro-Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"UAR:             "
        f"{uar:.4f}"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report_text = (
        classification_report(
            true_labels,
            predicted_labels,
            labels=list(
                range(len(labels))
            ),
            target_names=labels,
            zero_division=0,
        )
    )

    print(
        "\n=== CLASSIFICATION REPORT ==="
    )

    print(report_text)

    report_dict = (
        classification_report(
            true_labels,
            predicted_labels,
            labels=list(
                range(len(labels))
            ),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Metrics JSON
    # -----------------------------------------------------

    metrics = {
        "model": config[
            "model_name"
        ],
        "seed": config[
            "seed"
        ],
        "dataset_size": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "best_checkpoint":
            trainer.state
            .best_model_checkpoint,
        "best_validation_macro_f1":
            trainer.state.best_metric,
        "test_accuracy":
            accuracy,
        "test_macro_f1":
            macro_f1,
        "test_macro_precision":
            macro_precision,
        "test_uar":
            uar,
        "classification_report":
            report_dict,
    }

    with open(
        RESULT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # Save test probabilities
    # -----------------------------------------------------

    prediction_df = (
        test_metadata.copy()
    )

    prediction_df[
        "true_label"
    ] = [
        id2label[int(x)]
        for x in true_labels
    ]

    prediction_df[
        "predicted_label"
    ] = [
        id2label[int(x)]
        for x in predicted_labels
    ]

    for idx, label in enumerate(
        labels
    ):

        prediction_df[
            f"prob_{label}"
        ] = probabilities[:, idx]

    prediction_df.to_csv(
        RESULT_DIR
        / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(
            range(len(labels))
        ),
    )

    plt.figure(
        figsize=(9, 8)
    )

    plt.imshow(cm)

    plt.title(
        "WavLM - M3ED Test "
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(labels)),
        labels,
    )

    for i in range(
        len(labels)
    ):
        for j in range(
            len(labels)
        ):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR
        / "confusion_matrix.png",
        dpi=200,
    )

    plt.close()

    # -----------------------------------------------------
    # Training history
    # -----------------------------------------------------

    with open(
        RESULT_DIR
        / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            trainer.state.log_history,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Final paths
    # -----------------------------------------------------

    print(
        "\n=== SAVED ==="
    )

    print(
        "Best model:",
        best_model_dir,
    )

    print(
        "Metrics:",
        RESULT_DIR
        / "metrics.json",
    )

    print(
        "Predictions:",
        RESULT_DIR
        / "predictions.csv",
    )

    print(
        "Confusion matrix:",
        RESULT_DIR
        / "confusion_matrix.png",
    )


if __name__ == "__main__":
    main()