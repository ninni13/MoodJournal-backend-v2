import json
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TEXT_MODEL_DIR = (
    ROOT / "models/text_macbert_base/best_model"
)

SPEECH_MODEL_DIR = (
    ROOT / "models/speech_wavlm_base_plus/best_model"
)

FUSION_MODEL_PATH = (
    ROOT
    / "results/learned_fusion_logreg_balanced"
    / "fusion_model.joblib"
)

TEXT_CONFIG_PATH = (
    ROOT / "configs/text_macbert.json"
)

SPEECH_CONFIG_PATH = (
    ROOT / "configs/speech_wavlm.json"
)


# =========================================================
# Config
# =========================================================

with open(
    TEXT_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:
    text_config = json.load(f)

with open(
    SPEECH_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:
    speech_config = json.load(f)

LABELS = text_config["labels"]

if LABELS != speech_config["labels"]:
    raise RuntimeError(
        "Text and speech label order mismatch."
    )


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)


# =========================================================
# Load models once
# =========================================================

print("Loading MacBERT...")

text_tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL_DIR
)

text_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        TEXT_MODEL_DIR
    )
)

text_model.to(DEVICE)
text_model.eval()


print("Loading WavLM...")

speech_feature_extractor = (
    AutoFeatureExtractor.from_pretrained(
        SPEECH_MODEL_DIR
    )
)

speech_model = (
    AutoModelForAudioClassification
    .from_pretrained(
        SPEECH_MODEL_DIR
    )
)

speech_model.to(DEVICE)
speech_model.eval()


print("Loading fusion model...")

fusion_model = joblib.load(
    FUSION_MODEL_PATH
)

print("All models loaded.")


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="MoodJournal Emotion API",
    version="2.0.0",
)


# Development only.
# Later we can restrict this to your Vercel domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "MoodJournal-v2",
        "device": str(DEVICE),
        "labels": LABELS,
    }


# =========================================================
# Text inference
# =========================================================

def predict_text(text: str):

    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=text_config[
            "max_length"
        ],
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        logits = text_model(
            **inputs
        ).logits

        probs = torch.softmax(
            logits.float(),
            dim=-1,
        )[0]

    return probs.cpu().numpy()


# =========================================================
# Audio conversion
# =========================================================

def convert_to_wav(
    input_path: Path,
    output_path: Path,
):

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),

        # mono
        "-ac",
        "1",

        # 16 kHz
        "-ar",
        "16000",

        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr.decode(
                "utf-8",
                errors="ignore",
            )
        )


# =========================================================
# Speech inference
# =========================================================

def predict_speech(
    wav_path: Path,
):

    audio, sr = sf.read(
        wav_path,
        dtype="float32",
    )

    if sr != 16000:
        raise ValueError(
            f"Expected 16000 Hz, got {sr}"
        )

    if len(audio) == 0:
        raise ValueError(
            "Audio is empty."
        )

    inputs = (
        speech_feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        logits = speech_model(
            **inputs
        ).logits

        probs = torch.softmax(
            logits.float(),
            dim=-1,
        )[0]

    return probs.cpu().numpy()


# =========================================================
# Helpers
# =========================================================

def probs_to_dict(
    probabilities,
):

    return {
        label: float(
            probabilities[i]
        )
        for i, label
        in enumerate(LABELS)
    }


def top_label(
    probabilities,
):

    idx = int(
        np.argmax(
            probabilities
        )
    )

    return LABELS[idx]


# =========================================================
# Multimodal endpoint
# =========================================================

@app.post("/predict-fusion")
async def predict_fusion(
    text: str = Form(...),
    file: UploadFile | None = File(
        default=None
    ),
):

    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    try:

        # ---------------------------------------------
        # Text
        # ---------------------------------------------

        text_probs = predict_text(
            text
        )

        text_top1 = top_label(
            text_probs
        )

        # ---------------------------------------------
        # No audio -> Text only
        # ---------------------------------------------

        if (
            file is None
            or not file.filename
        ):

            return {
                "mode":
                    "text_only",

                "labels":
                    LABELS,

                "text_pred":
                    probs_to_dict(
                        text_probs
                    ),

                "audio_pred":
                    None,

                "fusion_pred":
                    probs_to_dict(
                        text_probs
                    ),

                "text_top1":
                    text_top1,

                "audio_top1":
                    None,

                "fusion_top1":
                    text_top1,

                "confidence":
                    float(
                        np.max(
                            text_probs
                        )
                    ),
            }

        # ---------------------------------------------
        # Audio
        # ---------------------------------------------

        audio_bytes = (
            await file.read()
        )

        if not audio_bytes:

            raise HTTPException(
                status_code=400,
                detail="Audio file is empty.",
            )

        with tempfile.TemporaryDirectory() as tmp:

            tmp_dir = Path(tmp)

            input_path = (
                tmp_dir / "input_audio"
            )

            wav_path = (
                tmp_dir / "audio.wav"
            )

            input_path.write_bytes(
                audio_bytes
            )

            convert_to_wav(
                input_path,
                wav_path,
            )

            speech_probs = (
                predict_speech(
                    wav_path
                )
            )

        speech_top1 = top_label(
            speech_probs
        )

        # ---------------------------------------------
        # Learned fusion
        #
        # 7 text probs + 7 speech probs = 14 features
        # ---------------------------------------------

        fusion_features = (
            np.concatenate(
                [
                    text_probs,
                    speech_probs,
                ]
            )
            .reshape(1, -1)
        )

        raw_fusion_probs = (
            fusion_model.predict_proba(
                fusion_features
            )[0]
        )

        # Make sure class probabilities are
        # restored to label index order.
        fusion_probs = np.zeros(
            len(LABELS),
            dtype=np.float32,
        )

        for class_id, prob in zip(
            fusion_model.classes_,
            raw_fusion_probs,
        ):

            fusion_probs[
                int(class_id)
            ] = prob

        fusion_top1 = top_label(
            fusion_probs
        )

        # ---------------------------------------------
        # Response
        # ---------------------------------------------

        return {
            "mode":
                "multimodal",

            "labels":
                LABELS,

            "text_pred":
                probs_to_dict(
                    text_probs
                ),

            "audio_pred":
                probs_to_dict(
                    speech_probs
                ),

            "fusion_pred":
                probs_to_dict(
                    fusion_probs
                ),

            "text_top1":
                text_top1,

            "audio_top1":
                speech_top1,

            "fusion_top1":
                fusion_top1,

            "confidence":
                float(
                    np.max(
                        fusion_probs
                    )
                ),
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "Prediction error:",
            repr(e),
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
