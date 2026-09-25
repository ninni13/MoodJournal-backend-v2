from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)


TEXT_VAL_PATH = Path(
    "results/text_macbert_base/val_predictions.csv"
)

SPEECH_VAL_PATH = Path(
    "results/speech_wavlm_base_plus/val_predictions.csv"
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
    # Load validation predictions
    # =====================================================

    text_df = pd.read_csv(TEXT_VAL_PATH)
    speech_df = pd.read_csv(SPEECH_VAL_PATH)

    print("=== INPUT ===")
    print("Text rows:", len(text_df))
    print("Speech rows:", len(speech_df))

    # =====================================================
    # Verify alignment
    # =====================================================

    if not text_df["id"].equals(speech_df["id"]):
        raise ValueError(
            "Text and speech validation IDs are not aligned."
        )

    if not text_df["label"].equals(speech_df["label"]):
        raise ValueError(
            "Text and speech true labels are not aligned."
        )

    print("ID alignment: OK")
    print("Label alignment: OK")

    # =====================================================
    # Probability columns
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

    label2id = {
        label: idx
        for idx, label in enumerate(LABELS)
    }

    y_true = (
        text_df["label"]
        .map(label2id)
        .to_numpy()
    )

    # =====================================================
    # Alpha search
    #
    # fusion =
    # alpha * text
    # + (1 - alpha) * speech
    # =====================================================

    results = []

    alphas = np.arange(
        0.0,
        1.0001,
        0.05,
    )

    print("\n=== ALPHA SEARCH ===")

    for alpha in alphas:

        fusion_probs = (
            alpha * text_probs
            + (1.0 - alpha)
            * speech_probs
        )

        y_pred = np.argmax(
            fusion_probs,
            axis=1,
        )

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

        uar = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        results.append(
            {
                "alpha_text": alpha,
                "alpha_speech": 1.0 - alpha,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "uar": uar,
            }
        )

        print(
            f"alpha={alpha:.2f} | "
            f"Accuracy={accuracy:.4f} | "
            f"Macro-F1={macro_f1:.4f} | "
            f"UAR={uar:.4f}"
        )

    # =====================================================
    # Results
    # =====================================================

    result_df = pd.DataFrame(results)

    # Best alpha chosen ONLY by validation Macro-F1
    best_row = result_df.loc[
        result_df["macro_f1"].idxmax()
    ]

    print("\n==============================")
    print("BEST VALIDATION ALPHA")
    print("==============================")

    print(
        f"Text alpha:   "
        f"{best_row['alpha_text']:.2f}"
    )

    print(
        f"Speech alpha: "
        f"{best_row['alpha_speech']:.2f}"
    )

    print(
        f"Accuracy:     "
        f"{best_row['accuracy']:.4f}"
    )

    print(
        f"Macro-F1:     "
        f"{best_row['macro_f1']:.4f}"
    )

    print(
        f"UAR:          "
        f"{best_row['uar']:.4f}"
    )

    # =====================================================
    # Save search results
    # =====================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_DIR / "alpha_search.csv",
        index=False,
    )

    best_row.to_frame().T.to_csv(
        OUTPUT_DIR / "best_alpha.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        OUTPUT_DIR / "alpha_search.csv"
    )
    print(
        OUTPUT_DIR / "best_alpha.csv"
    )


if __name__ == "__main__":
    main()
