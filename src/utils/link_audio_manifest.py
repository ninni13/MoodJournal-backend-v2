from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path("data/processed/manifest.csv")
BACKUP_PATH = Path("data/processed/manifest_before_audio.csv")

AUDIO_DIR = Path(
    "data/raw/M3ED_audio/modality_speech"
)


def main():
    df = pd.read_csv(MANIFEST_PATH)

    # Backup the manifest before modifying it
    if not BACKUP_PATH.exists():
        df.to_csv(
            BACKUP_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "Backup created:",
            BACKUP_PATH,
        )

    # Construct expected audio path:
    # e.g. B_sanshieryi_15_20.wav
    df["audio_path"] = df.apply(
        lambda row:
        str(
            AUDIO_DIR
            / f"{row['speaker']}_{row['id']}.wav"
        ),
        axis=1,
    )

    # Check existence
    exists = df["audio_path"].apply(
        lambda path: Path(path).is_file()
    )

    missing_df = df[~exists]

    print("\n=== AUDIO LINK CHECK ===")
    print("Manifest rows:", len(df))
    print("Audio found:", int(exists.sum()))
    print("Audio missing:", len(missing_df))

    if len(missing_df) > 0:
        print("\nMissing examples:")
        print(
            missing_df[
                ["id", "speaker", "audio_path"]
            ]
            .head(20)
            .to_string(index=False)
        )

        raise RuntimeError(
            "Missing audio detected. "
            "Manifest was not overwritten."
        )

    # Check duplicate audio paths
    duplicates = df[
        df["audio_path"].duplicated(
            keep=False
        )
    ]

    print(
        "Duplicate audio paths:",
        len(duplicates),
    )

    if len(duplicates) > 0:
        raise RuntimeError(
            "Duplicate audio mapping detected."
        )

    # Save canonical manifest
    df.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nManifest updated successfully.")
    print("Saved:", MANIFEST_PATH)

    print("\n=== FIRST 5 AUDIO PAIRS ===")

    print(
        df[
            [
                "id",
                "speaker",
                "text",
                "label",
                "audio_path",
            ]
        ]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
