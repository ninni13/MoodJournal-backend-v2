# MoodJournal-v2 — M3ED 情緒辨識實驗

使用 M3ED 的逐句文字與語音，比較 MacBERT、WavLM，以及三種 probability-level late fusion。任務為 **七類、單一標籤、utterance-level emotion classification**；標籤取自 `final_main_emo`，未使用影像、對話上下文或說話者特徵。

## 實驗成果

**重現狀態：** 已從原始資料完整重跑全部 13 步；MacBERT 數值完全一致，WavLM 重訓及其融合結果有差異。完整執行通過不代表逐位元重現，詳見重現檢查。

完整、可重新產生的 [Overall Results / 各 emotion Recall、F1 / 類別分布](results/summary/RESULTS.md)、[論文章節 Method / Experiment / Results / Discussion](docs/PAPER.md)、[實驗紀錄](EXPERIMENTS.md) 與 [重現檢查](docs/REPRODUCIBILITY.md)。

以下為原實驗的同一組 **4,198 筆 test utterances**；數值為百分比，seed = 42。

| Method | Accuracy | Macro-F1 | Macro-Precision | UAR |
| --- | ---: | ---: | ---: | ---: |
| MacBERT (text) | 45.07 | 30.43 | 33.50 | 29.13 |
| WavLM (speech) | 50.12 | 19.01 | 21.00 | 21.66 |
| Weighted fusion (α_text = 0.75) | 47.40 | 30.91 | **35.86** | 29.33 |
| Learned fusion (LR) | **53.26** | 30.38 | 35.43 | 29.91 |
| Balanced learned fusion (LR) | 41.40 | **34.14** | 35.19 | **36.28** |

UAR = Macro-Recall。Balanced fusion 在本次五個方法中 Macro-F1 / UAR 最高；普通 LR 的 Accuracy 最高。Fear 仍只辨識正確 5/65 筆，不能解讀為已解決少數類別問題。這些是單一 seed 的觀察結果，未做顯著性檢定。

## 環境

本機實際檢查使用 Linux、Python **3.10.21**、NVIDIA RTX 4090（24 GB）、driver **570.211.01**。

| 套件 | 版本 |
| --- | --- |
| PyTorch / torchaudio | 2.11.0+cu128 |
| Transformers | 5.17.0 |
| Accelerate | 1.15.0 |
| Datasets | 5.0.1 |
| NumPy / pandas | 2.2.6 / 2.3.3 |
| scikit-learn | 1.7.2 |
| matplotlib / soundfile | 3.10.9 / 0.14.0 |

`requirements.txt` 保留原環境的完整版本快照，包含 Linux/CUDA 相依套件。這不是跨作業系統的通用環境檔；本次沒有在全新環境重新安裝所有套件。

```bash
conda create -n moodjournal2-repro python=3.10.21 -y
conda activate moodjournal2-repro
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若原環境已存在，直接 `conda activate moodjournal2`。首次訓練需能下載 Hugging Face 預訓練模型，或已有完整本機快取。本次快取的 revisions 記於 `results/reproducibility/pretrained_revisions.json`；原設定未 pin revision。使用 CPU 可執行，但速度和浮點數行為不同；原實驗在支援時自動啟用 BF16。推論時 batch size、padding、精度與 GPU kernel 都應維持一致。實際版本及 GPU 資訊記錄於 `results/reproducibility/audit.json`。

## 資料準備

依 [隨資料附帶的說明](data/raw/M3ED_metadata/README.md) 取得 M3ED annotations、官方 split 檔與語音片段，再整理為：

```text
data/raw/
├── M3ED_metadata/
│   ├── annotation.json
│   └── splitInfo/
│       ├── movie_list_train.txt
│       ├── movie_list_val.txt
│       └── movie_list_test.txt
└── M3ED_audio/modality_speech/
    └── {speaker}_{utterance_id}.wav
```

原始標註共 **24,449** 筆。程式由官方 movie split 建立 manifest，再以 speaker 與 utterance ID 對應音檔。所有音檔為 16 kHz、單聲道；12 個檔案可讀但為空，排除 train 7、val 2、test 3 筆。最終 **24,437** 筆配對資料同時供文字、語音、融合使用，確保方法間樣本一致。程式沒有自動 resample、降噪、資料增強或語音辨識；文字直接使用標註 transcript。

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

固定類別順序：`Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise`。此順序不同於原資料 README 的 numeric label mapping；本專案從 label 名稱重新編碼，所有 `prob_*` 欄位按本專案順序使用。

## 如何執行

所有命令從專案根目錄執行。

### 已有實驗輸出：重新整理表格和圖片

```bash
python src/analysis/summarize_results.py
```

程式檢查 ID 唯一性、split coverage、true labels、機率合法性及 argmax，重新計算指標並核對原 metrics JSON；不一致即中止。輸出：

- `results/summary/RESULTS.md`：正式表格、各類 Recall/F1、類別分布、confusion matrix 圖。
- `overall_results.csv` / `overall_results.tex`：機器可讀與論文 LaTeX（需 `booktabs`）。
- `per_emotion_metrics.csv`：Precision、Recall、F1、Support、TP、FP、FN、預測數；另有 Recall/F1 寬表。
- `class_distribution.csv`：train/val/test 數量、比例與 balanced LR 權重。
- `confusion_matrices/`：每個方法及五方法合圖，counts / row-normalized，PNG、PDF、CSV。
- `provenance.json`：來源 SHA-256 與檢查範圍。CSV 的率為 0–1；Markdown 與圖中為百分比。

### 從原始資料完整訓練（保留原實驗）

```bash
python scripts/run_pipeline.py --output-dir runs/my-reproduction
```

`--output-dir` 必須尚不存在。入口會複製程式與設定、連結原始資料，依下節順序執行，模型與結果全部寫入新 run 目錄。進度印於終端，每一步完整 log 存於該目錄的 `logs/`；失敗會回傳非零 exit code。預訓練模型未快取時需要網路。

只檢查資料處理：

```bash
python scripts/run_pipeline.py --output-dir runs/data-check --stop-after data
```

### 驗證現有實驗是否可重現

```bash
# 原始資料重建 + saved probabilities 的完整融合重跑 + 報表驗算
python scripts/check_reproducibility.py --output results/audit-new

# 加上完整 val/test checkpoint 推論，以及原訓練程式的一個 epoch 小樣本測試
python scripts/check_reproducibility.py --models --output results/audit-models-new
```

Audit 在暫存目錄執行，不改動原模型或預測。`--output` 必須不存在或為空；輸出 JSON 與 logs。模型測試刻意離線，須已有兩個預訓練模型快取。它以零筆 label mismatch、機率絕對誤差 ≤ 1e-5 作為 replay 通過條件；差異會記錄為 `different` 並回傳 1，不會悄悄覆寫原實驗。**小樣本測試不等於完整重訓**；本次完整重跑結果另外記於 [重現檢查](docs/REPRODUCIBILITY.md)。

## 實驗順序與資料流

下表也是完整入口實際使用的順序。直接執行下列單一腳本會寫入目前工作目錄的固定 `data/`、`models/`、`results/` 路徑；保留既有實驗時請使用上面的新 run 入口。

| 順序 | 腳本 | 功能／輸出 |
| --- | --- | --- |
| 1 | `src/utils/build_manifest.py` | 標註與官方 split → `manifest.csv` |
| 2 | `src/utils/validate_manifest.py` | 檢查缺值、ID、label、movie/scene leakage；錯誤即中止 |
| 3 | `src/utils/link_audio_manifest.py` | 對應音檔，缺檔或重複路徑即中止 |
| 4 | `src/utils/validate_audio.py` | 音檔 metadata → `results/audio_validation.csv` |
| 5 | `src/utils/mark_audio_valid.py` | 排除空／不可讀音檔 → `manifest_multimodal.csv` |
| 6 | `src/text/train_text.py` | MacBERT 訓練、val checkpoint 選擇、test predictions |
| 7 | `src/speech/train_speech.py` | WavLM 訓練、val checkpoint 選擇、test predictions |
| 8 | `src/fusion/generate_val_predictions.py` | 兩個 best models → val probabilities |
| 9 | `src/fusion/search_weighted_fusion.py` | val Macro-F1 搜尋 α：0, 0.05, …, 1 |
| 10 | `src/fusion/evaluate_weighted_fusion.py` | 固定最佳 α，在 test 評估 |
| 11 | `src/fusion/train_learned_fusion.py` | val probabilities 訓練普通 LR，在 test 評估 |
| 12 | `src/fusion/train_learned_fusion_balanced.py` | val probabilities 訓練 balanced LR，在 test 評估 |
| 13 | `src/analysis/summarize_results.py` | 核對並整理本次 run 的表格與圖片 |

`generate_train_predictions.py` 是可選的診斷工具；這五個正式實驗的融合訓練 **不使用 train_predictions.csv**。融合用 validation 的 14 維機率特徵（7 text + 7 speech），不是 end-to-end joint training。

## 訓練設定與評估原則

設定檔為 `configs/text_macbert.json`、`configs/speech_wavlm.json`。

| 設定 | MacBERT | WavLM |
| --- | --- | --- |
| Pretrained ID | `hfl/chinese-macbert-base` | `microsoft/wavlm-base-plus` |
| 輸入 | max 32 tokens，dynamic padding | 原長 16 kHz waveform，dynamic padding |
| Learning rate | 2e-5 | 1e-5 |
| Train / eval batch size | 16 / 32 | 8 / 16 |
| Max epochs | 5 | 5 |
| Weight decay / seed | 0.01 / 42 | 0.01 / 42 |
| Checkpoint criterion | validation Macro-F1 | validation Macro-F1 |
| Early stopping patience | 2 epochs | 2 epochs |

原保存參數為 fused AdamW、linear LR scheduler、warmup steps 0、gradient accumulation 1、gradient clipping 1.0；使用 pinned Transformers 的預設值。兩個基礎模型均未加 class weights。Balanced LR 僅對融合訓練損失加權：`w_c = N_val / (7 × n_val,c)`；不重新抽樣，也不使用 test 頻率。

Validation 同時用於基礎模型 checkpoint 選擇、α 選擇／LR fitting；因此不能把 validation fitting score 當成獨立泛化表現。Test labels 未進入目前程式的 training / α search，但目前只有單 seed，且歷史上是否根據 test 結果調整設計無法從保存檔判定。報告以 Macro-F1 與 UAR 描述類別平衡表現，Accuracy 和 Weighted-F1 作補充。

## 專案結構

```text
configs/                 訓練設定
data/raw/                原資料；不由程式下載
data/processed/          原實驗 manifest
src/utils/               資料處理與驗證
src/text/, src/speech/   基礎模型訓練
src/fusion/              預測與融合
src/analysis/            結果整理與 checkpoint test replay
scripts/                 隔離重跑與重現稽核
models/                  原實驗 checkpoints / best_model
results/                 原實驗 predictions、metrics、融合模型
results/summary/         統一表格與 confusion matrices
docs/                    論文章節與重現檢查說明
runs/                    完整重新訓練的獨立輸出
```
