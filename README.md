# MoodJournal-v2 — M3ED 多模態情緒辨識

MoodJournal-v2 是一個以 **M3ED 中文多模態情緒資料集**為基礎的七類情緒辨識專案。

本專案使用逐句文字與語音，比較：

- MacBERT 文字模型
- WavLM 語音模型
- Weighted Late Fusion
- Learned Fusion
- Balanced Learned Fusion

任務定義為：

> **7-class, single-label, utterance-level emotion classification**

情緒標籤取自 M3ED 的 `final_main_emo`：

```text
Anger
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

目前未使用影像、對話上下文或說話者特徵。

除離線模型實驗外，本專案亦實作完整的互動式情緒日記系統。使用者可透過 React 前端輸入文字或錄製語音，FastAPI 後端會載入 MacBERT、WavLM 與 Balanced Learned Fusion 模型，回傳七類情緒機率與最終預測結果。

---

## Repositories

### Frontend

React + Vite 情緒日記介面：

[https://github.com/ninni13/MoodJournal](https://github.com/ninni13/MoodJournal)

主要功能包括：

- Firebase 使用者登入
- 情緒日記新增與管理
- 文字情緒分析
- 語音錄製
- 文字 + 語音 multimodal emotion recognition
- 七類情緒機率顯示
- 七類情緒統計與視覺化

### Backend / Training

本 repository：

```text
MoodJournal-v2
```

負責：

- M3ED 資料處理
- MacBERT 訓練
- WavLM 訓練
- Fusion 訓練與評估
- 實驗結果整理
- FastAPI inference service

### Model Weights

部署用模型儲存在 Hugging Face：

[https://huggingface.co/ninni13/moodjournal-multimodal-emotion](https://huggingface.co/ninni13/moodjournal-multimodal-emotion)

目前 repository 為 private。

包含：

```text
text_model/
speech_model/
fusion_model.joblib
labels.json
text_macbert.json
speech_wavlm.json
```

---

# 系統架構

```text
                    MoodJournal
                  React + Vite
                       │
                       │
              text + optional audio
                       │
                       ▼
                FastAPI Backend
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
       MacBERT                    WavLM
        Text                      Speech
          │                         │
          │ 7 probabilities         │ 7 probabilities
          └────────────┬────────────┘
                       │
                       ▼
             14-dimensional feature
                       │
                       ▼
        Balanced Logistic Regression
                       │
                       ▼
             7-class Emotion Output
                       │
      ┌────────────────────────────────┐
      │ Anger                          │
      │ Disgust                        │
      │ Fear                           │
      │ Happy                          │
      │ Neutral                        │
      │ Sad                            │
      │ Surprise                       │
      └────────────────────────────────┘
```

Fusion 並非 end-to-end multimodal network。

MacBERT 與 WavLM 各自輸出七維 probability vector，再串接為：

```text
7 text probabilities
+
7 speech probabilities
=
14-dimensional fusion feature
```

最後交由 Logistic Regression 進行情緒分類。

---

# 實驗成果

以下為原實驗相同的 **4,198 筆 test utterances**。

數值為百分比，seed = 42。

| Method | Accuracy | Macro-F1 | Macro-Precision | UAR |
| --- | ---: | ---: | ---: | ---: |
| MacBERT (text) | 45.07 | 30.43 | 33.50 | 29.13 |
| WavLM (speech) | 50.12 | 19.01 | 21.00 | 21.66 |
| Weighted Fusion (α_text = 0.75) | 47.40 | 30.91 | **35.86** | 29.33 |
| Learned Fusion (LR) | **53.26** | 30.38 | 35.43 | 29.91 |
| Balanced Learned Fusion (LR) | 41.40 | **34.14** | 35.19 | **36.28** |

其中：

- **Learned Fusion** 的 Accuracy 最高：53.26%
- **Balanced Learned Fusion** 的 Macro-F1 最高：34.14%
- **Balanced Learned Fusion** 的 UAR 最高：36.28%

UAR 即 Macro-Recall。

由於 M3ED 類別分布高度不平衡，本專案主要使用 **Macro-F1 與 UAR** 評估跨類別表現，而非僅依賴 Accuracy。

Balanced Fusion 改善了多數少數類別的 recall，但仍不能視為已完全解決 class imbalance。

例如 Fear 在 test set 中僅有 65 筆，其中目前只正確辨識 5 筆。

目前結果為 single-seed experiment，未進行統計顯著性檢定。

完整研究整理可參考：

- [論文章節整理：Method / Experiment / Results / Discussion](docs/PAPER.md)
- [實驗紀錄](EXPERIMENTS.md)
- [重現性檢查](docs/REPRODUCIBILITY.md)

完整 metrics、predictions、confusion matrices 與分析輸出不提交至 GitHub，可透過本 repository 的分析腳本重新產生至本機 `results/`。

---

# Dataset

本專案使用：

**M3ED — Multi-modal Multi-scene Multi-label Emotional Dialogue Dataset**

使用的主要資訊：

- transcript
- speech waveform
- `final_main_emo`
- 官方 movie-level train / validation / test split

沒有使用：

- video
- face
- speaker embedding
- dialogue context

---

## 資料數量

原始 annotation：

```text
24,449 utterances
```

其中有 12 個 WAV 檔案為可讀但沒有實際音訊內容的空檔，因此排除：

```text
Train       7
Validation  2
Test        3
```

最終可同時使用文字與語音的 paired dataset：

```text
24,437 utterances
```

分布如下：

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

---

## Label Order

本專案固定使用：

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

此順序不同於原資料 README 中的 numeric label mapping。

本專案是從 emotion label name 重新編碼，因此所有：

```text
prob_*
```

欄位皆遵循本 repository 的七類順序。

---

# Data Preparation

M3ED 資料本身不包含於本 repository。

取得 M3ED annotations、官方 split 與語音資料後，本機目錄結構應類似：

```text
data/raw/
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

所有有效音檔為：

```text
16 kHz
mono
WAV
```

本專案沒有自動執行：

- resampling
- denoising
- speech enhancement
- data augmentation
- automatic speech recognition

文字輸入直接使用 M3ED annotation 中的 transcript。

---

# Models

## MacBERT — Text Model

Pretrained model：

```text
hfl/chinese-macbert-base
```

用途：

```text
Chinese text emotion classification
```

主要設定：

| Setting | Value |
| --- | --- |
| Maximum length | 32 tokens |
| Padding | Dynamic |
| Learning rate | 2e-5 |
| Train batch size | 16 |
| Evaluation batch size | 32 |
| Maximum epochs | 5 |
| Weight decay | 0.01 |
| Seed | 42 |
| Checkpoint criterion | Validation Macro-F1 |
| Early stopping patience | 2 epochs |

---

## WavLM — Speech Model

Pretrained model：

```text
microsoft/wavlm-base-plus
```

輸入：

```text
16 kHz raw waveform
```

主要設定：

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

MacBERT 與 WavLM baseline 均未使用 class weights。

---

# Fusion Methods

本專案比較三種 probability-level late fusion。

## 1. Weighted Fusion

使用：

```text
P = α × P_text + (1 - α) × P_speech
```

在 validation set 搜尋：

```text
α = 0.00, 0.05, 0.10, ..., 1.00
```

選擇 validation Macro-F1 最高的 α。

最終：

```text
α_text = 0.75
α_speech = 0.25
```

---

## 2. Learned Fusion

將兩個模型的 probability vectors 串接：

```text
MacBERT: 7
WavLM:   7
----------------
Total:  14 features
```

使用 Logistic Regression 學習融合規則。

Fusion model 使用 validation predictions 進行 fitting。

---

## 3. Balanced Learned Fusion

架構與 Learned Fusion 相同，但 Logistic Regression 使用：

```python
class_weight="balanced"
```

其權重概念為：

```text
w_c = N / (K × n_c)
```

其中：

```text
N   = validation samples
K   = number of classes
n_c = samples of class c
```

Balanced Fusion 不重新抽樣資料，也不使用 test class frequencies。

目前 FastAPI deployment 使用的即為此模型。

---

# Training Pipeline

完整 pipeline：

| Step | Script | Function |
| --- | --- | --- |
| 1 | `src/utils/build_manifest.py` | 建立 manifest |
| 2 | `src/utils/validate_manifest.py` | 檢查資料與 split |
| 3 | `src/utils/link_audio_manifest.py` | 對應 speech files |
| 4 | `src/utils/validate_audio.py` | 驗證 audio metadata |
| 5 | `src/utils/mark_audio_valid.py` | 排除無效音檔 |
| 6 | `src/text/train_text.py` | MacBERT |
| 7 | `src/speech/train_speech.py` | WavLM |
| 8 | `src/fusion/generate_val_predictions.py` | 建立 validation probabilities |
| 9 | `src/fusion/search_weighted_fusion.py` | 搜尋 Weighted Fusion α |
| 10 | `src/fusion/evaluate_weighted_fusion.py` | Weighted Fusion test |
| 11 | `src/fusion/train_learned_fusion.py` | Learned Fusion |
| 12 | `src/fusion/train_learned_fusion_balanced.py` | Balanced Fusion |
| 13 | `src/analysis/summarize_results.py` | 統整實驗結果 |

`generate_train_predictions.py` 為額外診斷工具。

目前正式 Fusion 實驗：

```text
不使用 train_predictions.csv 來 fit fusion model
```

避免直接使用 base model 對自身 training set 的 in-sample predictions 作為正式融合結果。

---

# 完整重新訓練

所有 command 從 repository root 執行。

完整 pipeline：

```bash
python scripts/run_pipeline.py \
  --output-dir runs/my-reproduction
```

`--output-dir` 必須為尚未存在的目錄。

每個 run 會：

- 複製設定
- 保留 logs
- 重新處理資料
- 訓練 MacBERT
- 訓練 WavLM
- 執行 Fusion
- 統整結果

模型與輸出不會直接覆蓋原始實驗。

---

## 只檢查資料處理

```bash
python scripts/run_pipeline.py \
  --output-dir runs/data-check \
  --stop-after data
```

---

# Reproducibility Check

驗證資料與已儲存 probabilities：

```bash
python scripts/check_reproducibility.py \
  --output results/audit-new
```

加入 checkpoint inference：

```bash
python scripts/check_reproducibility.py \
  --models \
  --output results/audit-models-new
```

Audit 會在新的輸出目錄執行，不覆寫原始模型或結果。

檢查項目包括：

- label consistency
- probability validity
- split coverage
- prediction consistency
- checkpoint replay

完整說明：

[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)

---

# FastAPI Inference Service

Backend entry point：

```text
src/api/app.py
```

啟動時會從 Hugging Face model repository 載入：

```text
text_model/
speech_model/
fusion_model.joblib
```

---

## Hugging Face Authentication

目前 model repository 為 private。

部署環境需提供具有 read permission 的 Hugging Face token：

```bash
export HF_TOKEN=hf_xxxxxxxxx
```

Token 不應：

- 寫入 Python source code
- commit 到 GitHub
- 寫進 README
- 公開分享

本機若已執行：

```bash
hf auth login
```

Hugging Face Hub library 亦可使用本機保存的登入資訊。

---

# 啟動 Backend

進入環境：

```bash
conda activate moodjournal2
cd ~/MoodJournal-v2
```

啟動：

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

Text-only mode 使用 MacBERT probability output。

---

## Text + Speech Inference

```bash
curl -X POST \
  -F 'text=今天心情不太好' \
  -F 'file=@example.wav' \
  http://127.0.0.1:8000/predict-fusion
```

Backend 會：

```text
audio
  ↓
FFmpeg
  ↓
16 kHz mono WAV
  ↓
WavLM
```

再將：

```text
MacBERT probabilities
+
WavLM probabilities
```

送入 Balanced Learned Fusion。

---

# API Response

範例：

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
  "text_pred": {},
  "audio_pred": {},
  "fusion_pred": {},
  "text_top1": "Neutral",
  "audio_top1": "Neutral",
  "fusion_top1": "Sad",
  "confidence": 0.24
}
```

其中：

```text
text_pred
```

為 MacBERT 七類 probabilities。

```text
audio_pred
```

為 WavLM 七類 probabilities。

```text
fusion_pred
```

為最終 fusion 七類 probabilities。

---

# Frontend Connection

Frontend repository：

[https://github.com/ninni13/MoodJournal](https://github.com/ninni13/MoodJournal)

Frontend 透過：

```text
VITE_GATEWAY_BASE
```

指定 FastAPI endpoint。

Local development example：

```env
VITE_GATEWAY_BASE=http://127.0.0.1:8000
```

Frontend 使用：

```text
POST /predict-fusion
```

傳送：

```text
text
optional audio file
```

並顯示：

- 文字模型 probabilities
- 語音模型 probabilities
- Fusion probabilities
- 最終七類情緒 prediction

---

# Environment

原實驗環境：

```text
Linux
Python 3.10.21
NVIDIA RTX 4090 24 GB
NVIDIA Driver 570.211.01
CUDA 12.8
```

主要套件：

| Package | Version |
| --- | --- |
| PyTorch | 2.11.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| Transformers | 5.17.0 |
| Accelerate | 1.15.0 |
| Datasets | 5.0.1 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| soundfile | 0.14.0 |

`requirements.txt` 保留原環境的完整 package snapshot。

其中包含 Linux / CUDA-specific dependencies，因此不是保證可跨平台直接安裝的通用 requirements。

---

# 建立 Conda Environment

```bash
conda create -n moodjournal2-repro python=3.10.21 -y
conda activate moodjournal2-repro
```

接著：

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt
```

確認：

```bash
python -m pip check
```

GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

原實驗環境若仍存在：

```bash
conda activate moodjournal2
```

即可直接使用。

---

# Experiment Notes

原保存參數包含：

```text
AdamW
linear learning-rate scheduler
warmup_steps = 0
gradient_accumulation_steps = 1
gradient clipping = 1.0
```

原實驗在支援時使用 BF16。

Validation set 同時被用於：

- MacBERT checkpoint selection
- WavLM checkpoint selection
- Weighted Fusion α selection
- Logistic Regression fitting

因此 validation fitting score 不應視為獨立泛化結果。

Test labels 未直接進入目前 training script 或 α search。

然而，由於本專案是在持續開發過程中比較多種模型設計，因此歷史上的人工設計決策可能已參考過 test 結果。若用於正式論文中的嚴格模型選擇，應進一步考慮獨立 final test set、cross-validation 或 out-of-fold stacking。

目前結果亦僅為 single-seed experiment。

---

# Project Structure

```text
MoodJournal-v2/
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
│   │   └── app.py
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
├── EXPERIMENTS.md
├── README.md
├── requirements.txt
└── .gitignore
```

以下資料存在於本機，但不提交 GitHub：

```text
data/
models/
results/
hf_upload/
runs/
```

用途：

```text
data/        M3ED 原始與處理後資料
models/      training checkpoints
results/     predictions、metrics、fusion models、figures
hf_upload/   Hugging Face upload staging files
runs/        reproduction experiment outputs
```

---

# Git / Data Policy

為避免 repository 過大，以及避免重新散布 M3ED 原始資料，本專案 GitHub 不包含：

```text
data/
models/
results/
hf_upload/
runs/
.env
.env.local
```

部署用模型由 Hugging Face Hub 管理。

原始 M3ED dataset 需依資料集提供者的取得方式與授權條件自行下載。

---

# Current Status

目前已完成：

- [x] M3ED metadata parsing
- [x] Official movie-level train / validation / test split
- [x] Audio validation
- [x] MacBERT text baseline
- [x] WavLM speech baseline
- [x] Weighted Fusion
- [x] Learned Fusion
- [x] Balanced Learned Fusion
- [x] Evaluation and confusion matrices
- [x] Reproducibility pipeline
- [x] FastAPI inference backend
- [x] Hugging Face model storage
- [x] React frontend integration
- [x] Firebase user system
- [x] Browser audio recording
- [x] Text + speech multimodal inference
- [x] Seven-class emotion visualization

---

# Final Model Used by the Application

目前 MoodJournal application 的 multimodal inference 使用：

```text
Text:
hfl/chinese-macbert-base
        ↓
fine-tuned MacBERT

Speech:
microsoft/wavlm-base-plus
        ↓
fine-tuned WavLM

Fusion:
Balanced Logistic Regression
```

最終輸出：

```text
Anger
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

本專案的主要研究觀察為：

> 在高度不平衡的七類 M3ED emotion classification 中，單純提高整體 Accuracy 不一定能改善少數類別辨識；加入 balanced fusion 後，雖然整體 Accuracy 降低，但 Macro-F1 與 UAR 均有所提升。
