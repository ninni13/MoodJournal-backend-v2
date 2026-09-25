from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf


MANIFEST_PATH = Path("data/processed/manifest.csv")
OUTPUT_PATH = Path("results/audio_validation.csv")


def main():
    df = pd.read_csv(MANIFEST_PATH)

    results = []

    broken_files = []

    print("Checking audio files...")

    for i, row in df.iterrows():
        path = Path(row["audio_path"])

        try:
            info = sf.info(path)

            results.append(
                {
                    "id": row["id"],
                    "audio_path": str(path),
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "frames": info.frames,
                    "duration": info.duration,
                    "format": info.format,
                    "subtype": info.subtype,
                }
            )

        except Exception as e:
            broken_files.append(
                {
                    "id": row["id"],
                    "audio_path": str(path),
                    "error": str(e),
                }
            )

        if (i + 1) % 5000 == 0:
            print(f"Checked {i + 1}/{len(df)}")

    result_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== BASIC ===")
    print("Manifest:", len(df))
    print("Readable audio:", len(result_df))
    print("Broken audio:", len(broken_files))

    print("\n=== SAMPLE RATE ===")
    print(
        result_df["sample_rate"]
        .value_counts()
        .sort_index()
    )

    print("\n=== CHANNELS ===")
    print(
        result_df["channels"]
        .value_counts()
        .sort_index()
    )

    print("\n=== DURATION (seconds) ===")
    print(result_df["duration"].describe())

    print("\nShortest 10:")
    print(
        result_df.nsmallest(
            10,
            "duration",
        )[
            ["id", "duration", "sample_rate", "channels"]
        ].to_string(index=False)
    )

    print("\nLongest 10:")
    print(
        result_df.nlargest(
            10,
            "duration",
        )[
            ["id", "duration", "sample_rate", "channels"]
        ].to_string(index=False)
    )

    print("\nZero / invalid duration:")
    print(
        (result_df["duration"] <= 0).sum()
    )

    if broken_files:
        print("\n=== BROKEN FILES ===")

        for item in broken_files[:20]:
            print(item)

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
