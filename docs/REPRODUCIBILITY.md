# 重現檢查紀錄

本次把「資料／指標可重算」、「原程式可完整執行」與「獨立 checkpoint replay 的數值相同」分開檢查。原實驗的模型、predictions 和 metrics 沒有被重跑覆寫。

## 環境與來源

檢查使用原本 `moodjournal2` 環境：Python 3.10.21、PyTorch 2.11.0+cu128、Transformers 5.17.0、scikit-learn 1.7.2；RTX 4090，driver 570.211.01。`python -m pip check` 通過。完整套件快照為 `requirements.txt`，本次未從零建立新環境或驗證所有 wheel 的重新下載。

- [初次隔離 audit（JSON）](../results/reproducibility/audit.json)：版本、各步驟結果、時間、程式 SHA-256；同目錄保留 logs。
- [原結果來源與指標驗算](../results/summary/provenance.json)：逐筆預測、原 metrics JSON、manifest 的 SHA-256。
- 完整重新訓練輸出：`runs/reproduction-20260925/`；每一步程式與 config 快照、模型、結果、logs 均保留在該目錄。

## 1. 原始資料與既有機率的重現

以 `python scripts/check_reproducibility.py --models` 在暫存目錄執行，結果如下：

| 檢查 | 結果 |
| --- | --- |
| 從 annotation 與官方 split 重建 manifest | 通過；24,449 筆 |
| 音檔對應、metadata 掃描、空音檔排除 | 通過；12 個空檔，保留 24,437 筆 |
| 與原 full/paired manifest、音檔 metadata 比較 | 通過；逐欄一致 |
| 21 個 α 搜尋點與最佳 α | 通過；原始機率重算一致，最佳 0.75 |
| 三種融合的所有 test predictions | 通過；4,198 筆，機率容許誤差 1e-12 |
| 三種融合 metrics JSON | 通過；相同 |
| 五方法重新計算及 confusion-matrix 獨立公式核對 | 通過；原 JSON 與計算差異 < 1e-12 |
| 原 MacBERT / WavLM 訓練程式小樣本執行 | 通過；各 14 train、7 val、7 test，1 epoch，含存檔與 test 評估 |

Audit 的融合重現使用保存的基礎模型 probabilities，與下一節獨立 checkpoint replay 是兩種不同檢查；不能用前者的成功掩蓋後者的差異。

## 2. 獨立載入 checkpoint 的完整推論

原 `generate_val_predictions.py` 完整重跑 validation；新增 `src/analysis/replay_test_predictions.py` 使用已存 best_model、原 test batch size 與 BF16 重跑 test。判定條件是 labels 全部相同且機率最大絕對誤差 ≤ 1e-5。

| 模型／split | 樣本數 | Label 不同筆數 | 最大機率絕對差 | 判定 |
| --- | ---: | ---: | ---: | --- |
| MacBERT validation | 2,819 | 0 | 0 | 通過 |
| WavLM validation | 2,819 | 0 | 0 | 通過 |
| WavLM test | 4,198 | 0 | 0 | 通過 |
| MacBERT test | 4,198 | 8 | 0.0208962 | **未達逐筆 replay 一致** |

因此初次 audit 的 exit code 為 1，JSON 的 `different` 是真實發現，不是忽略的程式錯誤。FP32 診斷並未消除差異（MacBERT 15 筆、WavLM 10 筆 labels 不同）；原 tokenizer 與保存 tokenizer 在全部 4,198 筆 test 的 token IDs 一致。現有證據不足以把差異歸因於某一個 kernel 或 precision 設定，不能宣稱根因已修復。

原 test predictions 來自 `trainer.train()` 結束後的 `trainer.predict()`；獨立載入 checkpoint 是另一個執行路徑。Validation 也使用不同於 Trainer test 的 FP32 / batch-size 設定。若未來統一推論程序，應建立新的版本化實驗，避免直接混入原結果表。

## 3. 從頭完整重跑

執行命令：

```bash
python scripts/run_pipeline.py --output-dir runs/reproduction-20260925
```

入口依 README 的 13 個步驟執行，從原始 annotations/audio 開始，兩個 baseline 都重新從預訓練模型訓練，再產生 validation predictions、訓練／評估三種 fusion 與產出 summary。此 run 未使用原 fine-tuned weights 當訓練初始化。

**13 個步驟全部完成，完整流程 exit code = 0；數值完全一致檢查未通過（比較工具 exit code = 1）。** 兩個 baseline 均重新訓練 5 epochs。

| Method | Accuracy 原 / 重跑 (%) | Macro-F1 原 / 重跑 (%) | UAR 原 / 重跑 (%) |
| --- | ---: | ---: | ---: |
| MacBERT (text) | 45.07 / 45.07 | 30.43 / 30.43 | 29.13 / 29.13 |
| WavLM (speech) | 50.12 / 50.52 | 19.01 / 18.60 | 21.66 / 22.20 |
| Weighted fusion | 47.40 / 45.07 | 30.91 / 30.43 | 29.33 / 29.13 |
| Learned fusion (LR) | 53.26 / 53.31 | 30.38 / 30.16 | 29.91 / 29.77 |
| Balanced learned fusion (LR) | 41.40 / 41.64 | 34.14 / 34.08 | 36.28 / 36.41 |

MacBERT 的 validation/test probabilities、test labels 與 best-model weights SHA-256 完全一致。資料處理結果亦完全一致。WavLM 的權重與預測不同：原本選 epoch 5 / step 10,890，重跑選 epoch 4 / step 8,712；validation 最佳 Macro-F1 從 19.0066% 變為 18.2938%。WavLM 的 test labels 有 322/4,198 筆不同，validation 有 267/2,819 筆不同。加權融合選到 α_text = 1.00（原為 0.75），因此此次 weighted fusion 等於 text-only。

這表示固定 seed=42 本身不足以保證此 WavLM GPU 訓練逐位元一致。原程式沒有啟用 deterministic algorithms，也未完整保存原程序的 backend runtime flags；目前無法僅憑一次重跑確定根因。不得將不同 GPU kernel、浮點精度等可能原因寫成已證實的結論。Balanced LR 在兩次執行仍有最高 Macro-F1/UAR，但這只是兩次同 seed 執行的觀察，不是多 seed 穩健性證據。

正式表格仍呈現原始五組實驗。重跑輸出另存，不替換或挑選較好的分數。若要求完全 deterministic 的新實驗，需要固定 backend／推論設定、驗證支援的 deterministic operations，再重新建立整組 baseline；不能回溯宣稱本次已做到。

比較命令（可重跑）：

```bash
python scripts/compare_runs.py \
  --run runs/reproduction-20260925 \
  --output results/reproducibility
```

[完整重跑逐檔比較](../results/reproducibility/full_run_comparison.json) 包含 CSV 的 exact parsed-value 比較、label mismatch、機率最大差、best-model weights SHA-256，以及 run 的 source/config hashes；[指標差異表](../results/reproducibility/full_run_metrics_comparison.csv) 以 percentage points 呈現變化。

修正 manifest 的失敗條件並加入 replay 差異明細輸出後，再次執行的 [最終 audit](../results/reproducibility/final_audit/audit.json) 保留相同判定標準；MacBERT test 的不同 ID 列於 [差異明細](../results/reproducibility/final_audit/text_macbert_base_test_replay_mismatches.csv)。

## 4. 程式與文件檢查

- 新增完整執行入口，要求全新的 output directory；每步保存 log、遇到非零 exit code 即停止。
- `validate_manifest.py` 原本僅印出完整性問題；已加入實際失敗條件，避免壞資料繼續進入訓練。這不改變既有資料或模型方法。
- 報表輸入測試涵蓋重新排序的 ID 正確對齊，以及重複／缺失 ID、錯誤 true label、NaN／負值／非正規化機率、錯誤 argmax 的拒絕。
- `python -m unittest discover -s tests -v`：3 個測試方法及其中的 corruption cases 通過。
- `python -m compileall -q src scripts tests`：通過。
- `python -m pip check`：通過。
- 重複 manifest ID 與既有 output directory 的負向測試通過，10 個 confusion-matrix CSV 的列和／總數核對通過，見 [final_checks.json](../results/reproducibility/final_checks.json)。
- Counts 與 row-normalized 五方法圖已視覺檢查；PNG/PDF 採相同繪圖流程，CSV 保留完整數值。
- `overall_results.tex` 已用 `pdflatex` 搭配 `booktabs` 編譯通過。

## 可重現範圍

結果表可以由保存的逐筆預測確定性地重建，資料處理與融合可在原環境重跑。完整重訓的實際比較見上節。獨立 MacBERT test replay 仍有上述差異，需保留此限制；不要把「流程執行成功」等同於所有可能推論路徑都 bitwise identical。跨硬體／CUDA／套件版本、全新環境安裝與多 seed 的穩定性尚未驗證。

預訓練模型使用本機快取，對應 revision 記於 [pretrained_revisions.json](../results/reproducibility/pretrained_revisions.json)。原設定僅存 model ID，沒有 pin revision；跨時間重新下載的等同性尚未驗證。
