from pathlib import Path

import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt
import joblib


TEXT_VAL_PATH = Path(
    "results/text_macbert_base/val_predictions.csv"
)

SPEECH_VAL_PATH = Path(
    "results/speech_wavlm_base_plus/val_predictions.csv"
)

TEXT_TEST_PATH = Path(
    "results/text_macbert_base/predictions.csv"
)

SPEECH_TEST_PATH = Path(
    "results/speech_wavlm_base_plus/predictions.csv"
)

OUTPUT_DIR = Path(
    "results/learned_fusion_logreg_balanced"
)

LABELS = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


def prepare_features(
    text_df,
    speech_df,
    true_label_col,
):

    if not text_df["id"].equals(
        speech_df["id"]
    ):
        raise ValueError(
            "Text and speech IDs are not aligned."
        )

    if not text_df[
        true_label_col
    ].equals(
        speech_df[true_label_col]
    ):
        raise ValueError(
            "True labels are not aligned."
        )

    prob_cols = [
        f"prob_{label}"
        for label in LABELS
    ]

    text_probs = (
        text_df[prob_cols]
        .to_numpy()
    )

    speech_probs = (
        speech_df[prob_cols]
        .to_numpy()
    )

    # 14-dimensional feature:
    # 7 text probabilities
    # + 7 speech probabilities
    X = np.concatenate(
        [
            text_probs,
            speech_probs,
        ],
        axis=1,
    )

    label2id = {
        label: idx
        for idx, label
        in enumerate(LABELS)
    }

    y = (
        text_df[true_label_col]
        .map(label2id)
        .to_numpy()
    )

    return X, y


def main():

    # ==============================================
    # Load validation data
    # ==============================================

    text_val = pd.read_csv(
        TEXT_VAL_PATH
    )

    speech_val = pd.read_csv(
        SPEECH_VAL_PATH
    )

    print("=== FUSION TRAINING DATA ===")
    print(
        "Validation rows:",
        len(text_val),
    )

    X_train, y_train = (
        prepare_features(
            text_val,
            speech_val,
            "label",
        )
    )

    print(
        "Fusion feature shape:",
        X_train.shape,
    )

    # ==============================================
    # Learned Fusion
    # ==============================================

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
	class_weight="balanced",
    )

    print(
        "\nTraining learned fusion..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # ==============================================
    # Load test data
    # ==============================================

    text_test = pd.read_csv(
        TEXT_TEST_PATH
    )

    speech_test = pd.read_csv(
        SPEECH_TEST_PATH
    )

    print("\n=== TEST DATA ===")
    print(
        "Test rows:",
        len(text_test),
    )

    X_test, y_test = (
        prepare_features(
            text_test,
            speech_test,
            "true_label",
        )
    )

    # ==============================================
    # Test prediction
    # ==============================================

    y_pred = model.predict(
        X_test
    )

    test_probs = model.predict_proba(
        X_test
    )

    # ==============================================
    # Metrics
    # ==============================================

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    uar = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print(
        "\n=== TEST RESULTS ==="
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

    print(
        "\n=== CLASSIFICATION REPORT ==="
    )

    report_text = (
        classification_report(
            y_test,
            y_pred,
            labels=list(
                range(len(LABELS))
            ),
            target_names=LABELS,
            zero_division=0,
        )
    )

    print(report_text)

    report_dict = (
        classification_report(
            y_test,
            y_pred,
            labels=list(
                range(len(LABELS))
            ),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )
    )

    # ==============================================
    # Save
    # ==============================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        OUTPUT_DIR
        / "fusion_model.joblib",
    )

    id2label = {
        idx: label
        for idx, label
        in enumerate(LABELS)
    }

    result_df = pd.DataFrame(
        {
            "id":
                text_test["id"],
            "true_label":
                text_test["true_label"],
            "predicted_label":
                [
                    id2label[int(x)]
                    for x in y_pred
                ],
        }
    )

    for idx, label in enumerate(
        LABELS
    ):
        result_df[
            f"prob_{label}"
        ] = test_probs[:, idx]

    result_df.to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "method":
            "learned_probability_fusion_logistic_regression",
        "fusion_training_samples":
            len(X_train),
        "test_samples":
            len(X_test),
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
        OUTPUT_DIR
        / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ==============================================
    # Confusion matrix
    # ==============================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(
            range(len(LABELS))
        ),
    )

    plt.figure(
        figsize=(9, 8)
    )

    plt.imshow(cm)

    plt.title(
        "Learned Fusion - "
        "Test Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(
        range(len(LABELS)),
        LABELS,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(LABELS)),
        LABELS,
    )

    for i in range(
        len(LABELS)
    ):
        for j in range(
            len(LABELS)
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
        OUTPUT_DIR
        / "test_confusion_matrix.png",
        dpi=200,
    )

    plt.close()

    print("\n=== SAVED ===")

    print(
        OUTPUT_DIR
        / "fusion_model.joblib"
    )

    print(
        OUTPUT_DIR
        / "test_metrics.json"
    )

    print(
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    print(
        OUTPUT_DIR
        / "test_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
