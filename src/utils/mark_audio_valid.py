from pathlib import Path

import pandas as pd
import soundfile as sf


MANIFEST_PATH = Path("data/processed/manifest.csv")
MULTIMODAL_PATH = Path(
    "data/processed/manifest_multimodal.csv"
)


def check_audio(path):
    try:
        info = sf.info(path)

        return (
            info.frames > 0
            and info.duration > 0
        )

    except Exception:
        return False


def main():
    df = pd.read_csv(MANIFEST_PATH)

    print("Checking audio validity...")

    df["audio_valid"] = df["audio_path"].apply(
        check_audio
    )

    print("\n=== AUDIO VALIDITY ===")
    print(df["audio_valid"].value_counts())

    invalid = df[~df["audio_valid"]]

    print("\nInvalid audio:", len(invalid))

    print("\n=== INVALID BY SPLIT ===")
    print(invalid["split"].value_counts())

    print("\n=== INVALID BY LABEL ===")
    print(invalid["label"].value_counts())

    # Save full manifest with flag
    df.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Create fair multimodal subset
    multimodal_df = df[
        df["audio_valid"]
    ].copy()

    multimodal_df.to_csv(
        MULTIMODAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== MULTIMODAL DATASET ===")
    print("Total:", len(multimodal_df))

    print("\nSplit counts:")
    print(
        multimodal_df["split"].value_counts()
    )

    print("\nLabel counts:")
    print(
        multimodal_df["label"].value_counts()
    )

    print("\nSaved:")
    print(MANIFEST_PATH)
    print(MULTIMODAL_PATH)


if __name__ == "__main__":
    main()
