from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path("data/processed/manifest.csv")

EXPECTED_LABELS = {
    "Happy",
    "Neutral",
    "Sad",
    "Disgust",
    "Anger",
    "Fear",
    "Surprise",
}

EXPECTED_SPLITS = {"train", "val", "test"}


def main():
    df = pd.read_csv(MANIFEST_PATH)

    print("=== BASIC INFO ===")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # 1. ID uniqueness
    print("\n=== ID CHECK ===")
    duplicate_ids = df[df["id"].duplicated(keep=False)]

    print("Duplicate IDs:", len(duplicate_ids))

    if len(duplicate_ids) > 0:
        print(duplicate_ids.head(20).to_string(index=False))

    # 2. Missing values
    print("\n=== MISSING VALUES ===")

    required_columns = [
        "id",
        "movie",
        "scene_id",
        "speaker",
        "text",
        "label",
        "split",
        "start_time",
        "end_time",
    ]

    print(df[required_columns].isnull().sum())

    # 3. Empty text
    print("\n=== EMPTY TEXT ===")

    empty_text = (
        df["text"].isnull()
        | (df["text"].astype(str).str.strip() == "")
    )

    print("Empty text rows:", empty_text.sum())

    # 4. Labels
    print("\n=== LABEL CHECK ===")

    actual_labels = set(df["label"].dropna().unique())

    print("Labels:", sorted(actual_labels))
    print("Missing expected labels:", EXPECTED_LABELS - actual_labels)
    print("Unexpected labels:", actual_labels - EXPECTED_LABELS)

    # 5. Splits
    print("\n=== SPLIT CHECK ===")

    actual_splits = set(df["split"].dropna().unique())

    print("Splits:", sorted(actual_splits))
    print("Unexpected splits:", actual_splits - EXPECTED_SPLITS)

    # 6. Movie leakage
    print("\n=== MOVIE LEAKAGE CHECK ===")

    train_movies = set(df[df["split"] == "train"]["movie"])
    val_movies = set(df[df["split"] == "val"]["movie"])
    test_movies = set(df[df["split"] == "test"]["movie"])

    print("Train movies:", len(train_movies))
    print("Val movies:", len(val_movies))
    print("Test movies:", len(test_movies))

    print("Train ∩ Val:", train_movies & val_movies)
    print("Train ∩ Test:", train_movies & test_movies)
    print("Val ∩ Test:", val_movies & test_movies)

    # 7. Scene leakage
    print("\n=== SCENE LEAKAGE CHECK ===")

    scene_split_counts = df.groupby("scene_id")["split"].nunique()

    leaked_scenes = scene_split_counts[scene_split_counts > 1]

    print("Scenes appearing in multiple splits:", len(leaked_scenes))

    # 8. Speaker name overlap
    print("\n=== SPEAKER NAME OVERLAP ===")

    train_speakers = set(
        df[df["split"] == "train"]["speaker_name"].dropna()
    )
    val_speakers = set(
        df[df["split"] == "val"]["speaker_name"].dropna()
    )
    test_speakers = set(
        df[df["split"] == "test"]["speaker_name"].dropna()
    )

    print(
        "Train ∩ Val speaker names:",
        len(train_speakers & val_speakers),
    )

    print(
        "Train ∩ Test speaker names:",
        len(train_speakers & test_speakers),
    )

    print(
        "Val ∩ Test speaker names:",
        len(val_speakers & test_speakers),
    )

    # 9. Counts
    print("\n=== FINAL COUNTS ===")

    print("\nSplit:")
    print(df["split"].value_counts())

    print("\nLabel:")
    print(df["label"].value_counts())

    # A validation command must fail the pipeline when integrity checks fail.
    failures = []
    if not duplicate_ids.empty:
        failures.append("duplicate IDs")
    if df[required_columns].isnull().any().any():
        failures.append("missing required values")
    if empty_text.any():
        failures.append("empty text")
    if actual_labels != EXPECTED_LABELS:
        failures.append("unexpected or missing emotion classes")
    if actual_splits != EXPECTED_SPLITS:
        failures.append("unexpected or missing splits")
    if train_movies & val_movies or train_movies & test_movies or val_movies & test_movies:
        failures.append("movie leakage")
    if not leaked_scenes.empty:
        failures.append("scene leakage")
    if failures:
        raise ValueError("Manifest validation failed: " + "; ".join(failures))
    print("\nValidation finished: all required checks passed.")


if __name__ == "__main__":
    main()
