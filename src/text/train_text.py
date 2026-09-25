import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =========================================================
# Paths
# =========================================================

CONFIG_PATH = Path("configs/text_macbert.json")
MANIFEST_PATH = Path(
    "data/processed/manifest_multimodal.csv"
)
MODEL_DIR = Path("models/text_macbert_base")
RESULT_DIR = Path("results/text_macbert_base")


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Metrics
# =========================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_precision": precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Load config
    # -----------------------------------------------------

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    set_seed(config["seed"])

    labels = config["labels"]

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    print("=== CONFIG ===")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    print("\n=== LABEL MAPPING ===")
    print(label2id)

    # -----------------------------------------------------
    # Load manifest
    # -----------------------------------------------------

    df = pd.read_csv(MANIFEST_PATH)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("\n=== DATASET ===")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    # -----------------------------------------------------
    # Label encoding
    # -----------------------------------------------------

    for split_df in [train_df, val_df, test_df]:
        split_df["labels"] = split_df["label"].map(label2id)

        if split_df["labels"].isnull().any():
            raise ValueError("Unknown label found.")

        split_df["labels"] = split_df["labels"].astype(int)

    # Save test metadata before Hugging Face removes columns
    test_metadata = test_df[
        [
            "id",
            "movie",
            "scene_id",
            "speaker",
            "speaker_name",
            "text",
            "label",
        ]
    ].copy()

    # -----------------------------------------------------
    # Hugging Face Dataset
    # -----------------------------------------------------

    train_dataset = Dataset.from_pandas(
        train_df[["text", "labels"]],
        preserve_index=False,
    )

    val_dataset = Dataset.from_pandas(
        val_df[["text", "labels"]],
        preserve_index=False,
    )

    test_dataset = Dataset.from_pandas(
        test_df[["text", "labels"]],
        preserve_index=False,
    )

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"]
    )

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config["max_length"],
            padding=False,
        )

    print("Tokenizing datasets...")

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print("\nLoading MacBERT...")

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        label2id=label2id,
        id2label=id2label,
    )

    # -----------------------------------------------------
    # Training arguments
    # -----------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),

        learning_rate=config["learning_rate"],

        per_device_train_batch_size=
            config["train_batch_size"],

        per_device_eval_batch_size=
            config["eval_batch_size"],

        num_train_epochs=
            config["num_train_epochs"],

        weight_decay=
            config["weight_decay"],

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

        bf16=torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),

        report_to="none",
    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,

        processing_class=tokenizer,
        data_collator=data_collator,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=
                    config["early_stopping_patience"]
            )
        ],
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print("\n=================================")
    print("Starting MacBERT training")
    print("=================================\n")

    train_result = trainer.train()

    print("\nTraining finished.")

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

    best_model_dir = MODEL_DIR / "best_model"

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    print("\n=================================")
    print("Evaluating on TEST set")
    print("=================================\n")

    prediction_output = trainer.predict(
        tokenized_test
    )

    logits = prediction_output.predictions
    true_labels = prediction_output.label_ids

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

    macro_recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    print("=== TEST RESULTS ===")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro-F1:        {macro_f1:.4f}")
    print(f"Macro-Precision: {macro_precision:.4f}")
    print(f"Macro-Recall:    {macro_recall:.4f}")

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    metrics = {
        "model": config["model_name"],
        "seed": config["seed"],
        "max_length": config["max_length"],
        "best_checkpoint":
            trainer.state.best_model_checkpoint,
        "best_validation_macro_f1":
            trainer.state.best_metric,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_macro_precision": macro_precision,
        "test_macro_recall": macro_recall,
        "classification_report": report,
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
    # Save predictions
    # -----------------------------------------------------

    prediction_df = test_metadata.reset_index(
        drop=True
    )

    prediction_df["true_label"] = [
        id2label[int(x)]
        for x in true_labels
    ]

    prediction_df["predicted_label"] = [
        id2label[int(x)]
        for x in predicted_labels
    ]

    for idx, label in enumerate(labels):
        prediction_df[
            f"prob_{label}"
        ] = probabilities[:, idx]

    prediction_df.to_csv(
        RESULT_DIR / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(labels))),
    )

    plt.figure(figsize=(9, 8))

    plt.imshow(cm)

    plt.title(
        "MacBERT - M3ED Test Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

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

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "confusion_matrix.png",
        dpi=200,
    )

    plt.close()

    print("\n=== SAVED ===")
    print(
        "Best model:",
        best_model_dir,
    )
    print(
        "Metrics:",
        RESULT_DIR / "metrics.json",
    )
    print(
        "Predictions:",
        RESULT_DIR / "predictions.csv",
    )
    print(
        "Confusion matrix:",
        RESULT_DIR / "confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
