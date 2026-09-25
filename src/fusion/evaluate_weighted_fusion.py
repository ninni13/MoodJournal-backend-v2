from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


TEXT_TEST_PATH = Path(
    "results/text_macbert_base/predictions.csv"
)

SPEECH_TEST_PATH = Path(
    "results/speech_wavlm_base_plus/predictions.csv"
)

BEST_ALPHA_PATH = Path(
    "results/weighted_fusion/best_alpha.csv"
)

OUTPUT_DIR = Path(
    "results/weighted_fusion"
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


def main():

    # =====================================================
    # Load predictions
    # =====================================================

    text_df = pd.read_csv(
        TEXT_TEST_PATH
    )

    speech_df = pd.read_csv(
        SPEECH_TEST_PATH
    )

    print("=== TEST INPUT ===")
    print("Text rows:", len(text_df))
    print("Speech rows:", len(speech_df))

    # =====================================================
    # Verify alignment
    # =====================================================

    if not text_df["id"].equals(
        speech_df["id"]
    ):
        raise ValueError(
            "Text and speech test IDs "
            "are not aligned."
        )

    if not text_df["true_label"].equals(
        speech_df["true_label"]
    ):
        raise ValueError(
            "Text and speech true labels "
            "are not aligned."
        )

    print("ID alignment: OK")
    print("Label alignment: OK")

    # =====================================================
    # Load best validation alpha
    # =====================================================

    best_alpha_df = pd.read_csv(
        BEST_ALPHA_PATH
    )

    alpha_text = float(
        best_alpha_df.iloc[0][
            "alpha_text"
        ]
    )

    alpha_speech = float(
        best_alpha_df.iloc[0][
            "alpha_speech"
        ]
    )

    print("\n=== FIXED ALPHA ===")
    print(
        f"Text:   {alpha_text:.2f}"
    )
    print(
        f"Speech: {alpha_speech:.2f}"
    )

    # =====================================================
    # Probabilities
    # =====================================================

    prob_cols = [
        f"prob_{label}"
        for label in LABELS
    ]

    text_probs = text_df[
        prob_cols
    ].to_numpy()

    speech_probs = speech_df[
        prob_cols
    ].to_numpy()

    # =====================================================
    # Fusion
    # =====================================================

    fusion_probs = (
        alpha_text * text_probs
        + alpha_speech * speech_probs
    )

    y_pred = np.argmax(
        fusion_probs,
        axis=1,
    )

    label2id = {
        label: idx
        for idx, label
        in enumerate(LABELS)
    }

    id2label = {
        idx: label
        for label, idx
        in label2id.items()
    }

    y_true = (
        text_df["true_label"]
        .map(label2id)
        .to_numpy()
    )

    # =====================================================
    # Metrics
    # =====================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    uar = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\n=== TEST RESULTS ===")

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

    # =====================================================
    # Classification report
    # =====================================================

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(LABELS))
        ),
        target_names=LABELS,
        zero_division=0,
    )

    print(
        "\n=== CLASSIFICATION REPORT ==="
    )

    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(LABELS))
        ),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    # =====================================================
    # Save predictions
    # =====================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df = text_df[
        [
            "id",
            "movie",
            "scene_id",
            "speaker",
            "speaker_name",
            "text",
            "true_label",
        ]
    ].copy()

    result_df[
        "predicted_label"
    ] = [
        id2label[int(x)]
        for x in y_pred
    ]

    for idx, label in enumerate(
        LABELS
    ):

        result_df[
            f"prob_{label}"
        ] = fusion_probs[:, idx]

    result_df.to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # Save metrics
    # =====================================================

    metrics = {
        "method":
            "weighted_late_fusion",
        "text_alpha":
            alpha_text,
        "speech_alpha":
            alpha_speech,
        "alpha_selected_on":
            "validation_macro_f1",
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

    # =====================================================
    # Confusion matrix
    # =====================================================

    cm = confusion_matrix(
        y_true,
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
        "Weighted Late Fusion - "
        "M3ED Test Confusion Matrix"
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
