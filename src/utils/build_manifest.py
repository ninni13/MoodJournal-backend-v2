import json
from pathlib import Path

import pandas as pd


ANNOTATION_PATH = Path("data/raw/M3ED_metadata/annotation.json")
SPLIT_DIR = Path("data/raw/M3ED_metadata/splitInfo")
OUTPUT_PATH = Path("data/processed/manifest.csv")


def load_split_map():
    split_map = {}

    for split in ["train", "val", "test"]:
        path = SPLIT_DIR / f"movie_list_{split}.txt"

        with open(path, "r", encoding="utf-8") as f:
            movies = [line.strip() for line in f if line.strip()]

        for movie in movies:
            if movie in split_map:
                raise ValueError(f"{movie} appears in multiple splits")

            split_map[movie] = split

    return split_map


def main():
    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_map = load_split_map()

    rows = []

    for movie, scenes in data.items():

        if movie not in split_map:
            raise ValueError(f"Movie {movie} has no official split")

        split = split_map[movie]

        for scene_id, scene_data in scenes.items():

            dialog = scene_data["Dialog"]
            speaker_info = scene_data["SpeakerInfo"]

            for utterance_id, utterance in dialog.items():

                speaker = utterance["Speaker"]

                speaker_name = None
                if speaker in speaker_info:
                    speaker_name = speaker_info[speaker].get("Name")

                emotion = utterance["EmoAnnotation"]["final_main_emo"]

                rows.append(
                    {
                        "id": utterance_id,
                        "movie": movie,
                        "scene_id": scene_id,
                        "speaker": speaker,
                        "speaker_name": speaker_name,
                        "text": utterance["Text"],
                        "label": emotion,
                        "split": split,
                        "start_time": utterance["StartTime"],
                        "end_time": utterance["EndTime"],
                        "audio_path": "",
                    }
                )

    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Manifest created successfully")
    print("Path:", OUTPUT_PATH)
    print("Total utterances:", len(df))

    print("\n=== SPLIT COUNTS ===")
    print(df["split"].value_counts())

    print("\n=== LABEL COUNTS ===")
    print(df["label"].value_counts())

    print("\n=== LABEL COUNTS BY SPLIT ===")
    print(pd.crosstab(df["label"], df["split"]))

    print("\n=== FIRST 5 ROWS ===")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
