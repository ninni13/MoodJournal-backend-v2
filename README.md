# MoodJournal-backend-v2 — M3ED Multimodal Emotion Recognition

MoodJournal-backend-v2 is the backend, training, evaluation, and deployment repository for **MoodJournal**, a Chinese multimodal emotion diary application.

The project performs:

> **7-class, single-label, utterance-level emotion classification**

using **text and speech** from the M3ED dataset.

The system combines:

- **MacBERT** for Chinese text emotion recognition
- **WavLM** for speech emotion recognition
- **Probability-level late fusion**
- **Balanced Logistic Regression Fusion**
- **FastAPI** for model inference
- **Hugging Face Hub** for model storage
- **Google Cloud Run** for production backend deployment
- **React + Vite + Firebase** frontend deployed on Vercel

The seven emotion classes are:

```text
Anger
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

The current system does not use video, facial features, dialogue context, or speaker embeddings.

---

# System Overview

The complete production architecture is:

```text
User Browser
     │
     ▼
Vercel
React + Vite Frontend
     │
     │ HTTPS
     │ text + optional recorded audio
     ▼
Google Cloud Run
FastAPI Backend
     │
     ├───────────────┐
     │               │
     ▼               ▼
  MacBERT          WavLM
   Text            Speech
     │               │
     │ 7 probs       │ 7 probs
     └───────┬───────┘
             │
             ▼
    14-dimensional feature
             │
             ▼
Balanced Logistic Regression
             │
             ▼
      7-class prediction
             │
             ▼
Anger / Disgust / Fear /
Happy / Neutral / Sad / Surprise
```

The fusion model is **not an end-to-end multimodal neural network**.

Instead:

```text
MacBERT → 7 probabilities
WavLM   → 7 probabilities
              ↓
       concatenate
              ↓
        14 features
              ↓
 Logistic Regression
```

---

# Related Repositories

## Frontend

MoodJournal frontend:

```text
https://github.com/ninni13/MoodJournal
```

Production website:

```text
https://nis-moodjournal.vercel.app
```

Main frontend features include:

- Firebase authentication
- Emotion diary creation
- Text input
- Browser microphone recording
- Text-only emotion recognition
- Text + speech multimodal emotion recognition
- Seven-class emotion probability display
- Emotion chips
- Emotion distribution visualization
- Emotion calendar / diary insights
- Diary editing and trash management

---

## Backend / Training

This repository:

```text
https://github.com/ninni13/MoodJournal-backend-v2
```

This repository contains:

- M3ED preprocessing scripts
- Data validation
- MacBERT training
- WavLM training
- Fusion experiments
- Evaluation
- Reproducibility utilities
- FastAPI inference backend
- Docker / Cloud Run deployment configuration

---

## Model Repository

Deployment model weights are stored in a private Hugging Face model repository:

```text
ninni13/moodjournal-multimodal-emotion
```

Repository structure:

```text
text_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json

speech_model/
├── config.json
├── model.safetensors
└── preprocessor_config.json

fusion_model.joblib
labels.json
text_macbert.json
speech_wavlm.json
```

Model weights are intentionally not committed to GitHub.

---

# Task Definition

The project uses the M3ED `final_main_emo` field as the target label.

The task is:

```text
Input:
    text
    speech
        ↓
Output:
    one of seven emotion classes
```

The seven classes are fixed in this order:

```python
[
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]
```

All probability vectors in this project use the same order.

---

# Dataset

The project uses the **M3ED** Chinese multimodal emotion dataset.

The project uses:

- transcript
- speech waveform
- `final_main_emo`
- official movie-level train / validation / test split

The project does not currently use:

- video
- facial features
- dialogue context
- speaker embeddings

---

## Dataset Preparation

The original metadata contains:

```text
24,449 utterances
```

During speech validation, 12 WAV files were found to be readable files with zero effective duration.

Excluded samples:

```text
Train:       7
Validation:  2
Test:        3
```

Final paired text + speech dataset:

```text
24,437 utterances
```

Split sizes:

```text
Train:       17,420
Validation:   2,819
Test:         4,198
```

---

## Class Distribution

| Emotion | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Anger | 3,814 | 681 | 736 |
| Disgust | 1,145 | 134 | 218 |
| Fear | 280 | 50 | 65 |
| Happy | 1,625 | 303 | 358 |
| Neutral | 7,126 | 1,042 | 1,853 |
| Sad | 2,734 | 489 | 734 |
| Surprise | 696 | 120 | 234 |
| **Total** | **17,420** | **2,819** | **4,198** |

The dataset is strongly imbalanced, particularly for:

```text
Fear
Disgust
Surprise
```

Therefore, this project reports class-balanced metrics such as:

```text
Macro-F1
UAR / Macro-Recall
```

in addition to Accuracy.

---

# Local Dataset Structure

M3ED itself is not distributed through this repository.

Expected local structure:

```text
data/
└── raw/
    ├── M3ED_metadata/
    │   ├── annotation.json
    │   └── splitInfo/
    │       ├── movie_list_train.txt
    │       ├── movie_list_val.txt
    │       └── movie_list_test.txt
    │
    └── M3ED_audio/
        └── modality_speech/
            └── {speaker}_{utterance_id}.wav
```

Valid audio files are:

```text
16 kHz
mono
WAV
```

The current preprocessing pipeline does not automatically apply:

- denoising
- speech enhancement
- data augmentation
- ASR
- automatic resampling of the original dataset

Text input uses the annotated transcript directly.

---

# Models

# 1. MacBERT Text Model

Pretrained model:

```text
hfl/chinese-macbert-base
```

Task:

```text
Chinese utterance-level emotion classification
```

Main training settings:

| Setting | Value |
| --- | --- |
| Max sequence length | 32 tokens |
| Padding | Dynamic |
| Learning rate | 2e-5 |
| Train batch size | 16 |
| Evaluation batch size | 32 |
| Maximum epochs | 5 |
| Weight decay | 0.01 |
| Seed | 42 |
| Checkpoint criterion | Validation Macro-F1 |
| Early stopping patience | 2 epochs |

The baseline does not use class weighting.

---

# 2. WavLM Speech Model

Pretrained model:

```text
microsoft/wavlm-base-plus
```

Input:

```text
16 kHz raw waveform
```

Main training settings:

| Setting | Value |
| --- | --- |
| Learning rate | 1e-5 |
| Train batch size | 8 |
| Evaluation batch size | 16 |
| Maximum epochs | 5 |
| Weight decay | 0.01 |
| Seed | 42 |
| Checkpoint criterion | Validation Macro-F1 |
| Early stopping patience | 2 epochs |

The WavLM baseline also does not use class weighting.

---

# Fusion Methods

Three probability-level late-fusion approaches were evaluated.

---

## 1. Weighted Fusion

The probability vectors are combined using:

```text
P_fusion =
α × P_text
+
(1 - α) × P_speech
```

The text weight was searched on the validation set using:

```text
0.00
0.05
0.10
...
1.00
```

The selection criterion was:

```text
Validation Macro-F1
```

Best validation setting:

```text
α_text   = 0.75
α_speech = 0.25
```

---

## 2. Learned Fusion

The full probability vectors from both models are concatenated.

```text
MacBERT probabilities = 7
WavLM probabilities   = 7

Total features        = 14
```

A Logistic Regression classifier is then trained using these 14 features.

The fusion model is fitted using validation predictions.

---

## 3. Balanced Learned Fusion

Balanced Learned Fusion uses the same 14-dimensional feature representation but trains Logistic Regression with:

```python
class_weight="balanced"
```

Conceptually, the class weight is:

```text
w_c = N / (K × n_c)
```

where:

```text
N   = number of validation samples
K   = number of emotion classes
n_c = number of validation samples in class c
```

This approach:

- does not oversample
- does not undersample
- does not use test-set class frequencies

This is the fusion model currently used by the deployed application.

---

# Experiment Results

All methods below were evaluated on the same:

```text
4,198 test utterances
```

Seed:

```text
42
```

| Method | Accuracy | Macro-F1 | Macro-Precision | UAR |
| --- | ---: | ---: | ---: | ---: |
| MacBERT (Text) | 45.07 | 30.43 | 33.50 | 29.13 |
| WavLM (Speech) | 50.12 | 19.01 | 21.00 | 21.66 |
| Weighted Fusion (`α_text = 0.75`) | 47.40 | 30.91 | **35.86** | 29.33 |
| Learned Fusion | **53.26** | 30.38 | 35.43 | 29.91 |
| Balanced Learned Fusion | 41.40 | **34.14** | 35.19 | **36.28** |

---

## Main Observations

### Highest Accuracy

```text
Learned Fusion
Accuracy = 53.26%
```

### Highest Macro-F1

```text
Balanced Learned Fusion
Macro-F1 = 34.14%
```

### Highest UAR

```text
Balanced Learned Fusion
UAR = 36.28%
```

Balanced fusion sacrifices overall Accuracy but improves class-balanced performance.

This is relevant because M3ED is strongly imbalanced.

---

## Minority-Class Limitation

The improvement from balanced fusion does not mean that minority-class recognition is solved.

For example:

```text
Fear test support = 65
Correct Fear predictions = 5
```

Therefore, minority-class performance remains a major limitation of the current system.

---

# Metric Interpretation

The project reports:

```text
Accuracy
Macro-Precision
Macro-F1
UAR
```

UAR is equivalent to:

```text
Macro-Recall
```

Because the class distribution is imbalanced, the main class-balanced metrics used for interpretation are:

```text
Macro-F1
UAR
```

rather than Accuracy alone.

---

# Training Pipeline

The main training pipeline follows these steps:

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | `src/utils/build_manifest.py` | Build dataset manifest |
| 2 | `src/utils/validate_manifest.py` | Validate labels and split |
| 3 | `src/utils/link_audio_manifest.py` | Match audio files |
| 4 | `src/utils/validate_audio.py` | Validate audio metadata |
| 5 | `src/utils/mark_audio_valid.py` | Exclude invalid audio |
| 6 | `src/text/train_text.py` | Train MacBERT |
| 7 | `src/speech/train_speech.py` | Train WavLM |
| 8 | `src/fusion/generate_val_predictions.py` | Generate validation probabilities |
| 9 | `src/fusion/search_weighted_fusion.py` | Search weighted-fusion α |
| 10 | `src/fusion/evaluate_weighted_fusion.py` | Evaluate weighted fusion |
| 11 | `src/fusion/train_learned_fusion.py` | Train Logistic Regression fusion |
| 12 | `src/fusion/train_learned_fusion_balanced.py` | Train balanced Logistic Regression |
| 13 | `src/analysis/summarize_results.py` | Generate result summaries |

An additional:

```text
generate_train_predictions.py
```

script is available for diagnostic purposes.

The formal fusion experiments do **not** directly fit on `train_predictions.csv`.

---

# Full Reproduction

All commands should be executed from the repository root.

To run the complete training pipeline:

```bash
python scripts/run_pipeline.py \
  --output-dir runs/my-reproduction
```

The target output directory must not already exist.

Each run stores:

```text
logs
configs
models
predictions
metrics
analysis output
```

inside the isolated run directory.

---

## Data-only Check

To stop after preprocessing:

```bash
python scripts/run_pipeline.py \
  --output-dir runs/data-check \
  --stop-after data
```

---

# Reproducibility Check

To validate preprocessing and saved probability outputs:

```bash
python scripts/check_reproducibility.py \
  --output results/audit-new
```

To additionally replay model inference:

```bash
python scripts/check_reproducibility.py \
  --models \
  --output results/audit-models-new
```

The audit checks include:

- ID uniqueness
- split coverage
- label consistency
- probability validity
- argmax consistency
- saved prediction replay
- checkpoint inference replay

Additional notes are available in:

```text
docs/REPRODUCIBILITY.md
```

---

# FastAPI Inference Service

Backend entry point:

```text
src/api/app.py
```

The backend provides:

```text
GET  /health
POST /predict-fusion
```

---

# Model Loading

The backend downloads deployment models from the private Hugging Face repository using:

```python
snapshot_download(...)
```

Model paths:

```text
text_model/
speech_model/
fusion_model.joblib
```

The repository ID is:

```text
ninni13/moodjournal-multimodal-emotion
```

---

# Hugging Face Authentication

The application expects:

```text
HF_TOKEN
```

to be available as an environment variable.

The Python backend reads:

```python
HF_TOKEN = os.getenv("HF_TOKEN")
```

The token must not be:

- committed to GitHub
- hardcoded into Python files
- included in README
- exposed in frontend code

---

# Local Backend Execution

Activate the environment:

```bash
conda activate moodjournal2
cd ~/MoodJournal-v2
```

Start FastAPI:

```bash
uvicorn src.api.app:app \
  --host 127.0.0.1 \
  --port 8000
```

---

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

---

## Text-only Inference

```bash
curl -X POST \
  -F 'text=我今天真的很開心，發生了很多好事' \
  http://127.0.0.1:8000/predict-fusion
```

In text-only mode:

```text
MacBERT prediction
→ returned as text prediction
→ fusion output follows the text-only probabilities
```

No speech model is executed when no audio file is provided.

---

## Multimodal Inference

```bash
curl -X POST \
  -F 'text=今天心情不太好' \
  -F 'file=@example.wav' \
  http://127.0.0.1:8000/predict-fusion
```

Processing flow:

```text
text
  ↓
MacBERT

audio
  ↓
FFmpeg
  ↓
16 kHz mono WAV
  ↓
WavLM

MacBERT probabilities
+
WavLM probabilities
  ↓
Balanced Logistic Regression
  ↓
Final seven-class emotion prediction
```

---

# API Response

Example response structure:

```json
{
  "mode": "multimodal",
  "labels": [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
  ],
  "text_pred": {
    "Anger": 0.01,
    "Disgust": 0.01,
    "Fear": 0.01,
    "Happy": 0.45,
    "Neutral": 0.49,
    "Sad": 0.02,
    "Surprise": 0.01
  },
  "audio_pred": {
    "Anger": 0.03,
    "Disgust": 0.06,
    "Fear": 0.02,
    "Happy": 0.08,
    "Neutral": 0.49,
    "Sad": 0.28,
    "Surprise": 0.04
  },
  "fusion_pred": {
    "Anger": 0.03,
    "Disgust": 0.10,
    "Fear": 0.12,
    "Happy": 0.39,
    "Neutral": 0.15,
    "Sad": 0.13,
    "Surprise": 0.08
  },
  "text_top1": "Neutral",
  "audio_top1": "Neutral",
  "fusion_top1": "Happy",
  "confidence": 0.39
}
```

The final fusion class does not have to match either base model's top-1 class.

The Logistic Regression fusion classifier uses all 14 probabilities rather than performing simple majority voting.

---

# Audio Processing

Browser recordings may arrive in formats such as:

```text
WebM
```

Before WavLM inference, FastAPI uses FFmpeg to convert the recording into:

```text
16 kHz
mono
WAV
```

Temporary audio files are removed after inference.

---

# Frontend Integration

Frontend repository:

```text
https://github.com/ninni13/MoodJournal
```

Frontend environment variable:

```env
VITE_GATEWAY_BASE=https://<cloud-run-service-url>
```

Local development example:

```env
VITE_GATEWAY_BASE=http://127.0.0.1:8000
```

Vite environment variables are embedded during build time, so Vercel must be redeployed after changing:

```text
VITE_GATEWAY_BASE
```

---

# Production Deployment

The application is deployed using:

```text
Frontend:
Vercel

Backend:
Google Cloud Run

Model storage:
Hugging Face Hub

Secret storage:
Google Secret Manager
```

---

# Production Architecture

```text
                        Internet
                           │
                           ▼
               https://nis-moodjournal.vercel.app
                           │
                           │
                       Vercel
                   React + Vite
                           │
                           │ HTTPS
                           ▼
                  Google Cloud Run
                     FastAPI API
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
             MacBERT                WavLM
                │                     │
                └──────────┬──────────┘
                           │
                           ▼
              Balanced Logistic Regression
                           │
                           ▼
                  7-class emotion result
```

---

# Google Cloud Run Configuration

Current production settings:

```text
Service:
moodjournal-backend-v2

Region:
asia-east1 (Taiwan)

Billing:
Request-based

Authentication:
Public access enabled

Ingress:
All

CPU:
2 vCPU

Memory:
4 GiB

Minimum instances:
0

Maximum instances:
1

Concurrency:
1

Request timeout:
300 seconds

Container port:
8080

GPU:
Disabled
```

The service is intentionally configured with:

```text
Minimum instances = 0
```

so it can scale to zero when unused.

Maximum instances are limited to:

```text
1
```

to reduce unexpected costs during the portfolio/demo stage.

---

# Docker Deployment

Cloud Run builds the backend using:

```text
Dockerfile
```

The production Docker image uses CPU PyTorch rather than the local CUDA training environment.

Container start command:

```bash
uvicorn src.api.app:app \
  --host 0.0.0.0 \
  --port ${PORT:-8080}
```

Cloud Run provides the `PORT` environment variable automatically.

---

# Cloud Build

The Cloud Run service is connected to:

```text
GitHub
ninni13/MoodJournal-backend-v2
```

Production branch:

```text
main
```

Cloud Build automatically builds the Docker image used by Cloud Run.

This means future backend changes can follow:

```bash
git add .
git commit -m "Describe the change"
git push
```

and the configured deployment pipeline can rebuild the service.

---

# Google Secret Manager

The Hugging Face deployment token is not stored directly inside the repository.

Secret name:

```text
moodjournal-hf-token
```

It is exposed to the Cloud Run container as:

```text
HF_TOKEN
```

The Cloud Run runtime service account is granted:

```text
Secret Manager Secret Accessor
```

for this secret.

The actual token value is never committed to GitHub.

---

# Vercel Production Configuration

Production frontend:

```text
https://nis-moodjournal.vercel.app
```

The Vercel project contains:

```text
VITE_GATEWAY_BASE
```

which points to the Cloud Run backend.

After changing this value, the Vercel frontend must be redeployed.

---

# Production Verification

The deployed application has been verified using both:

```text
Text-only inference
```

and:

```text
Text + Speech multimodal inference
```

The production pipeline successfully performs:

```text
Browser recording
→ Vercel frontend
→ Cloud Run FastAPI
→ FFmpeg
→ MacBERT + WavLM
→ Balanced Fusion
→ Seven-class result
```

---

# Environment

Original experiment environment:

```text
OS:
Linux

Python:
3.10.21

GPU:
NVIDIA RTX 4090 24 GB

NVIDIA Driver:
570.211.01

CUDA:
12.8
```

Main package versions:

| Package | Version |
| --- | --- |
| PyTorch | 2.11.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| Transformers | 5.17.0 |
| Accelerate | 1.15.0 |
| Datasets | 5.0.1 |
| Evaluate | 0.4.6 |
| librosa | 0.11.0 |
| soundfile | 0.14.0 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |

`requirements.txt` represents the original training environment snapshot.

It contains Linux / CUDA-specific dependencies and is therefore not intended to be a universal cross-platform requirements file.

---

# Deployment Dependencies

Cloud Run uses the smaller deployment dependency file:

```text
requirements-deploy.txt
```

This avoids unnecessarily installing the full local training environment inside the production container.

Deployment dependencies include:

```text
fastapi
uvicorn
python-multipart
huggingface-hub
transformers
torch CPU
numpy
scikit-learn
joblib
librosa
soundfile
```

System packages include:

```text
ffmpeg
libsndfile1
```

---

# Conda Environment

To create a reproduction environment:

```bash
conda create \
  -n moodjournal2-repro \
  python=3.10.21 \
  -y
```

Activate:

```bash
conda activate moodjournal2-repro
```

Install:

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt
```

Check environment:

```bash
python -m pip check
```

Check CUDA:

```bash
python -c \
'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The original environment can be activated with:

```bash
conda activate moodjournal2
```

---

# Experiment Configuration

Saved training settings include:

```text
Optimizer:
AdamW

Scheduler:
Linear learning-rate scheduler

Warmup:
0 steps

Gradient accumulation:
1

Gradient clipping:
1.0

Seed:
42
```

BF16 was used when supported by the training environment.

---

# Experimental Design Notes

The validation set is used for multiple development decisions:

- MacBERT checkpoint selection
- WavLM checkpoint selection
- weighted-fusion α selection
- Logistic Regression fusion fitting

Therefore, validation performance must not be interpreted as an independent estimate of final generalization performance.

The final reported metrics use the test split.

However, this project was developed iteratively and multiple model variants were compared over time.

Therefore, historical development decisions may have been influenced by previously observed test performance.

For stricter future research evaluation, possible improvements include:

- a fully held-out final test set
- cross-validation
- out-of-fold stacking
- repeated seeds
- statistical significance testing

The current reported results are from a single seed.

---

# Project Structure

```text
MoodJournal-backend-v2/
│
├── configs/
│   ├── text_macbert.json
│   └── speech_wavlm.json
│
├── docs/
│   ├── PAPER.md
│   └── REPRODUCIBILITY.md
│
├── scripts/
│   ├── run_pipeline.py
│   └── check_reproducibility.py
│
├── src/
│   ├── api/
│   │   ├── app.py
│   │   ├── app_hf.py
│   │   └── app_local_backup.py
│   │
│   ├── analysis/
│   │
│   ├── fusion/
│   │
│   ├── speech/
│   │
│   ├── text/
│   │
│   └── utils/
│
├── tests/
│
├── Dockerfile
├── requirements.txt
├── requirements-deploy.txt
├── EXPERIMENTS.md
├── README.md
├── .dockerignore
└── .gitignore
```

Some API backup files may exist locally depending on development history.

---

# Local-only Files

The following directories are intentionally excluded from GitHub:

```text
data/
models/
results/
hf_upload/
runs/
```

Purpose:

```text
data/
    Original and processed M3ED data

models/
    Local training checkpoints

results/
    Predictions, metrics, fusion models and figures

hf_upload/
    Temporary Hugging Face upload staging directory

runs/
    Reproduction experiment output
```

---

# Git Ignore Policy

Sensitive or large files are excluded from source control.

Examples:

```text
data/
models/
results/
hf_upload/
runs/

.env
.env.local

__pycache__/
*.pyc

checkpoints/
```

The repository may include:

```text
.env.example
```

but must never contain real secrets.

---

# Data Policy

The original M3ED dataset is not committed to this repository.

Users must obtain M3ED through the dataset provider and follow the corresponding licensing and usage conditions.

This repository contains only code required to process the locally obtained dataset.

---

# Security Notes

Never commit:

```text
Hugging Face access tokens
Firebase private secrets
Google Cloud credentials
.env files containing credentials
```

Production secrets should be managed through:

```text
Google Secret Manager
Vercel Environment Variables
```

rather than source code.

---

# Current Project Status

Completed:

- [x] M3ED metadata parsing
- [x] Official movie-level split
- [x] Audio validation
- [x] Paired multimodal manifest
- [x] MacBERT text baseline
- [x] WavLM speech baseline
- [x] Weighted late fusion
- [x] Learned Logistic Regression fusion
- [x] Balanced Logistic Regression fusion
- [x] Macro-F1 / UAR evaluation
- [x] Confusion matrices
- [x] Reproducibility utilities
- [x] FastAPI inference API
- [x] Browser audio support
- [x] FFmpeg audio conversion
- [x] React frontend integration
- [x] Firebase authentication
- [x] Seven-class emotion display
- [x] Seven-class emotion visualization
- [x] Hugging Face model repository
- [x] GitHub backend repository
- [x] Docker deployment
- [x] Google Secret Manager integration
- [x] Google Cloud Run deployment
- [x] Vercel frontend deployment
- [x] Production text-only verification
- [x] Production multimodal verification

---

# Final Application Model

The deployed application currently uses:

```text
Text branch
    ↓
hfl/chinese-macbert-base
    ↓
Fine-tuned MacBERT
    ↓
7 probabilities
```

and:

```text
Speech branch
    ↓
microsoft/wavlm-base-plus
    ↓
Fine-tuned WavLM
    ↓
7 probabilities
```

followed by:

```text
14 probabilities
    ↓
Balanced Logistic Regression
    ↓
Final emotion
```

Output classes:

```text
Anger
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

---

# Main Project Finding

The current experiments illustrate that:

> Higher overall Accuracy does not necessarily correspond to better performance across all emotion classes in an imbalanced emotion-recognition task.

The ordinary learned fusion model achieved the highest overall Accuracy, while Balanced Learned Fusion reduced Accuracy but improved:

```text
Macro-F1
UAR
```

indicating better class-balanced performance.

At the same time, performance on very small classes such as Fear remains limited and should not be considered solved.

---

# Production Stack

```text
Frontend
React
Vite
Firebase
Vercel

Backend
Python
FastAPI
Uvicorn
FFmpeg
Google Cloud Run

Machine Learning
PyTorch
Transformers
MacBERT
WavLM
scikit-learn Logistic Regression

Model Storage
Hugging Face Hub

Secret Management
Google Secret Manager

Version Control
GitHub
```

---

# Final Deployment Flow

```text
User
 │
 ▼
https://nis-moodjournal.vercel.app
 │
 ▼
React + Vite
 │
 │ text / microphone recording
 ▼
Google Cloud Run
 │
 ▼
FastAPI
 │
 ├── MacBERT
 │
 └── WavLM
       │
       ▼
Balanced Fusion
       │
       ▼
7-class emotion output
       │
       ▼
Diary + emotion visualization
```

The complete MoodJournal multimodal emotion-recognition application is currently deployed and operational.