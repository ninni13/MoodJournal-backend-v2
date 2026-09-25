# M3ED Text–Speech Emotion Recognition: Experimental Manuscript

This manuscript describes the five saved experiments in this repository. All numerical claims refer to the original prediction files, not a newly selected run. The study uses one seed (42); differences are descriptive and are not claimed to be statistically significant. Full precision tables and source hashes are available in [the generated results](../results/summary/RESULTS.md).

## Method

### Task and input representation

We formulate emotion recognition as seven-class, single-label classification of individual utterances. The target is the M3ED annotation field `EmoAnnotation.final_main_emo`. The class order is Anger, Disgust, Fear, Happy, Neutral, Sad, and Surprise. Although the source dataset supports richer annotations, the present implementation uses only the final main emotion. Each input consists of an annotated Chinese transcript and its corresponding speech waveform. The models do not receive visual features, preceding or following turns, speaker identity, or movie identity; these metadata are retained only for alignment and evaluation.

### Text and speech baselines

The text branch fine-tunes `hfl/chinese-macbert-base` through `AutoModelForSequenceClassification` with a seven-output classification head. Text is tokenized with the model tokenizer, truncated to a maximum sequence length of 32 tokens including special tokens, and dynamically padded within each batch. The speech branch fine-tunes `microsoft/wavlm-base-plus` through `AutoModelForAudioClassification`. Audio is read as float32 at 16 kHz and passed to the saved feature extractor with dynamic padding and an attention mask. The saved extractor has `do_normalize=false`; the implementation does not apply additional waveform normalization, resampling, denoising, fixed-duration cropping, or data augmentation.

Both branches optimize the standard single-label cross-entropy loss without class weighting. The code does not freeze either backbone. Model outputs are converted to seven-dimensional probability vectors using softmax, denoted by \(p^{(t)}\) and \(p^{(s)}\) for text and speech, respectively. For each branch, the checkpoint with the highest validation Macro-F1 is used for subsequent evaluation and fusion.

### Weighted probability fusion

Weighted late fusion combines the branch probabilities as

\[
p^{(f)}_c = \alpha p^{(t)}_c + (1-\alpha)p^{(s)}_c,
\qquad \hat{y}=\arg\max_c p^{(f)}_c.
\]

We search 21 values, \(\alpha\in\{0,0.05,\ldots,1\}\), using validation Macro-F1 alone; the implementation selects the first maximum in ascending alpha order if there is a tie. The selected text weight is 0.75, with speech weight 0.25. The weight is fixed before the test set is evaluated.

### Learned probability fusion

Learned fusion concatenates the seven text probabilities and seven speech probabilities into a 14-dimensional feature vector:

\[
x=[p^{(t)}_1,\ldots,p^{(t)}_7,p^{(s)}_1,\ldots,p^{(s)}_7].
\]

A multiclass logistic regression model maps this vector to the final emotion. Both variants use scikit-learn `LogisticRegression(max_iter=1000, random_state=42)` with the installed version's default L2 regularization, \(C=1\), intercept, and L-BFGS solver. Features are neither standardized nor calibrated. The unweighted variant uses the empirical validation distribution; the balanced variant additionally sets `class_weight="balanced"`. Both classifiers are fitted on all 2,819 validation utterances. The base models remain fixed, and there is no joint fine-tuning.

For balanced fusion, the weight for class \(c\) is

\[
w_c = \frac{N_{val}}{7n_{val,c}}.
\]

Consequently, the Disgust, Fear, and Neutral weights are approximately 3.0053, 8.0543, and 0.3865. These weights are computed from the fusion training set (validation), not the base training set or test set. They modify the training objective rather than resampling examples or changing test labels.

## Experiment

### Dataset construction and split protocol

The manifest is reconstructed from the local M3ED annotations and the supplied official movie split files. Each waveform is linked using `{speaker}_{utterance_id}.wav`. Of 24,449 annotated utterances, 12 have readable but empty audio and are excluded: seven training, two validation, and three test utterances. The paired subset therefore contains 24,437 utterances: 17,420 train, 2,819 validation, and 4,198 test. The corresponding movie counts are 38, 7, and 11. Audits found unique utterance IDs and no overlapping movies or scenes across splits.

All five methods use this same paired subset, including the text-only baseline. This controls sample eligibility when comparing modalities. The construction should not be described as a random utterance split, and movie disjointness should not be interpreted as proof of speaker disjointness. Audio metadata checks confirm 16 kHz mono audio in all 24,449 files before empty-file exclusion.

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

### Optimization and experiment sequence

Both baselines use seed 42, weight decay 0.01, a maximum of five epochs, evaluation and checkpoint saving after each epoch, and early stopping patience of two evaluations. MacBERT uses a learning rate of 2e-5, training batch size 16, and evaluation batch size 32. WavLM uses 1e-5, 8, and 16, respectively. Saved training arguments record fused AdamW, a linear scheduler, zero warmup steps, gradient accumulation of one, and gradient norm clipping at 1.0. BF16 was enabled on the available GPU. Both original runs reached epoch five; the selected MacBERT checkpoint is step 4,356 (epoch four), while WavLM uses step 10,890 (epoch five). Their checkpoint-selection validation Macro-F1 scores are 30.99% and 19.01%.

The sequence is: (1) construct and validate the paired manifest; (2) train MacBERT; (3) train WavLM; (4) produce validation probabilities using both selected checkpoints; (5) search the weighted fusion coefficient; (6) evaluate fixed weighted fusion on test; (7) fit and evaluate unweighted LR; (8) fit and evaluate balanced LR. The generated `train_predictions.csv` files are not inputs to any of these fusion experiments.

The base checkpoints use validation labels for selection, and the same validation set is reused for alpha selection or LR fitting. It is therefore not an independent evaluation set for the fusion models. The test labels are used only for evaluation in the supplied implementation. Saved files do not establish whether historical experiment design decisions were influenced by prior test inspection.

### Metrics and computational environment

We report Accuracy, Macro-F1, Macro-Precision, and unweighted average recall (UAR). UAR is the arithmetic mean of the seven per-class recalls. Macro-F1 averages seven per-class F1 scores; it is not the harmonic mean of Macro-Precision and UAR. Undefined precision or F1 is assigned zero (`zero_division=0`). Support-weighted F1 is supplied as a secondary metric in the machine-readable table. Per-emotion Recall/F1 and both absolute and row-normalized confusion matrices accompany the overall comparison.

The audit environment is Linux with Python 3.10.21, PyTorch 2.11.0+cu128, Transformers 5.17.0, scikit-learn 1.7.2, and an NVIDIA RTX 4090 with driver 570.211.01. The complete package snapshot is in `requirements.txt`; this records the installed environment and is not evidence that a fresh installation has been tested. Validation probability generation uses a standalone FP32 inference path (text batch size 64, speech batch size 8), whereas the original test predictions use Trainer with BF16 and the evaluation batch sizes above. This precision/batching difference is part of the implemented protocol and is a reproducibility limitation.

## Results

### Overall performance

All values in the following table are percentages on the same 4,198 test utterances. Bold identifies the maximum in a column among the five methods.

| Method | Accuracy | Macro-F1 | Macro-Precision | UAR |
| --- | --- | --- | --- | --- |
| MacBERT (text) | 45.07 | 30.43 | 33.50 | 29.13 |
| WavLM (speech) | 50.12 | 19.01 | 21.00 | 21.66 |
| Weighted fusion | 47.40 | 30.91 | **35.86** | 29.33 |
| Learned fusion (LR) | **53.26** | 30.38 | 35.43 | 29.91 |
| Balanced learned fusion (LR) | 41.40 | **34.14** | 35.19 | **36.28** |

Balanced learned fusion achieves the highest Macro-F1 (34.14%) and UAR (36.28%), while unweighted learned fusion achieves the highest Accuracy (53.26%). Relative to MacBERT, weighted fusion improves Macro-F1 by 0.48 percentage points (pp), Accuracy by 2.33 pp, and UAR by 0.20 pp. Thus, its overall gain is small and should not be described as a uniform per-class improvement.

Relative to unweighted LR, balancing increases Macro-F1 by 3.77 pp and UAR by 6.37 pp, but decreases Accuracy by 11.86 pp. Relative to weighted fusion, the balanced model gains 3.23 pp Macro-F1 and 6.95 pp UAR, with a 6.00 pp Accuracy decrease. These comparisons describe a trade-off between empirical-frequency accuracy and equal-class performance.

### Per-emotion Recall and F1

Each cell below is **Recall / F1 (%)**; support is the number of true test examples. The full precision, false-positive, and prediction-count breakdown is in [per_emotion_metrics.csv](../results/summary/per_emotion_metrics.csv).

| Emotion | Support | MacBERT (text) | WavLM (speech) | Weighted fusion | Learned fusion (LR) | Balanced learned fusion (LR) |
| --- | --- | --- | --- | --- | --- | --- |
| Anger | 736 | 36.96 / 36.49 | 53.12 / 54.27 | 40.35 / 40.57 | 63.86 / 59.34 | 51.49 / 56.19 |
| Disgust | 218 | 13.30 / 13.09 | 0.00 / 0.00 | 11.93 / 12.97 | 0.00 / 0.00 | 37.61 / 17.54 |
| Fear | 65 | 3.08 / 4.82 | 0.00 / 0.00 | 3.08 / 5.19 | 0.00 / 0.00 | 7.69 / 7.19 |
| Happy | 358 | 18.44 / 22.53 | 4.75 / 7.38 | 16.20 / 21.32 | 11.17 / 16.99 | 32.68 / 26.26 |
| Neutral | 1853 | 67.19 / 60.44 | 90.07 / 65.06 | 72.69 / 62.53 | 79.11 / 65.68 | 42.20 / 51.23 |
| Sad | 734 | 25.20 / 29.04 | 3.68 / 6.38 | 23.43 / 28.31 | 26.16 / 30.62 | 36.10 / 33.82 |
| Surprise | 234 | 39.74 / 46.62 | 0.00 / 0.00 | 37.61 / 45.48 | 29.06 / 40.00 | 46.15 / 46.75 |

Anger attains its highest Recall/F1 with unweighted LR (63.86% / 59.34%). Neutral attains its highest Recall with WavLM (90.07%) and highest F1 with unweighted LR (65.68%). Balanced LR has the highest observed Recall and F1 for Disgust, Fear, Happy, Sad, and Surprise. Nevertheless, the Surprise F1 difference from MacBERT is only 0.14 pp, and no inference of significance is warranted. Relative to MacBERT, weighted fusion lowers Recall for Disgust, Happy, Sad, and Surprise, while Fear Recall is unchanged.

### Disgust and Fear under class imbalance

Fear is the smallest class in all three splits: 280 training examples (1.61%), 50 validation examples (1.77%), and 65 test examples (1.55%). Neutral has 7,126 training examples (40.91%), a Neutral-to-Fear ratio of 25.45:1. Disgust has 1,145 training examples (6.57%), 134 validation examples (4.75%), and 218 test examples (5.19%); its training Neutral-to-Disgust ratio is 6.22:1. Disgust is underrepresented relative to Neutral but is not the second-smallest training class; Surprise has only 696 training examples. The two difficult classes should therefore not be described as equally rare.

Unweighted LR and WavLM produce no Disgust or Fear predictions on this test set. Balanced LR increases Disgust true positives from zero to 82/218 (Recall 37.61%) and Fear true positives from zero to 5/65 (Recall 7.69%). This improvement comes with substantial false positives: 717 examples are predicted as Disgust, of which 635 are incorrect (Precision 11.44%); 74 are predicted as Fear, of which 69 are incorrect (Precision 6.76%). Their F1 values remain low, at 17.54% and 7.19%.

MacBERT and weighted fusion each correctly identify only two Fear examples. Balanced LR identifies three additional examples, so the apparent Recall increase from 3.08% to 7.69% is based on very few utterances. One correct Fear example changes its Recall by 1.54 pp; one correct Disgust example changes its Recall by approximately 0.46 pp. This sensitivity should accompany any discussion of minority-class improvements.

### Confusion matrices

![Row-normalized confusion matrices](../results/summary/confusion_matrices/all_normalized.png)

*Figure 1. Test confusion matrices with true emotions in rows and predicted emotions in columns. All panels use the same class order and a shared 0–100% scale. Row supports are shown explicitly, and diagonal entries equal class Recall. Percentages are rounded for display.*

The [absolute-count figure](../results/summary/confusion_matrices/all_counts.pdf), [normalized vector figure](../results/summary/confusion_matrices/all_normalized.pdf), individual method figures, and CSV matrices are retained for publication and numerical inspection. Count panels share a common scale across methods.

WavLM predicts Neutral for 3,278/4,198 test utterances (78.08%), despite Neutral accounting for 44.14% of the true test labels. It routes 178/218 Disgust examples (81.65%) and 50/65 Fear examples (76.92%) to Neutral, and never predicts Disgust, Fear, or Surprise. Unweighted LR also routes many Disgust examples to Neutral (141/218, 64.68%); its Fear errors are concentrated in Neutral (29/65, 44.62%) and Sad (28/65, 43.08%).

Balanced LR reduces these Neutral-directed errors but introduces others. Its Neutral Recall falls from the unweighted LR's 79.11% to 42.20%; 301 true Neutral examples are predicted as Disgust and 346 as Sad. Fear remains predominantly confused with Sad (29/65, 44.62%), while only five Fear examples lie on the diagonal. The matrices therefore support improved minority sensitivity, but not robust separation of all minority emotions.

## Discussion

### Accuracy and equal-class performance answer different questions

The test set is dominated by Neutral, and WavLM's high Neutral Recall contributes strongly to its 50.12% Accuracy. Its Macro-F1 of 19.01% reveals poor coverage of other emotions. As a diagnostic reference, always predicting Neutral would yield 44.14% Accuracy on this test set; this is a distribution-derived baseline, not an additional trained experiment. WavLM's higher Accuracy than MacBERT therefore does not imply stronger performance across all classes.

The balanced fusion objective gives each class greater relative influence during optimization. The observed rise in minority Recall, together with the fall in Neutral Recall and overall Accuracy, is consistent with this changed objective. However, the experiments do not establish class imbalance as the sole cause of Disgust/Fear errors. Limited feature separability, label ambiguity, movie-domain differences, transcript truncation, and the absence of conversational context are plausible contributing factors that were not isolated by controlled ablations.

### What fusion contributes and what it does not

The two modalities provide useful but uneven signals in this implementation. Weighted fusion modestly improves aggregate results over text alone, while unweighted LR favors strong Anger/Neutral decisions and yields the best Accuracy. Balancing the LR loss changes this allocation of predictions and produces the best equal-class metrics. Since the learned classifier receives only probability vectors from fixed branches, it cannot recover acoustic or linguistic information already lost by those branches. Larger minority-class weights may also amplify ambiguous inputs, as demonstrated by the low Disgust/Fear precision.

### Limitations and follow-up experiments

This study contains one seed and one official split; it provides no confidence intervals, repeated-seed variability, or statistical significance tests. Utterances within the same scene or movie are dependent, so any future uncertainty analysis should respect that grouping rather than assume all utterances are independent. The 65-example Fear test support is particularly small.

The validation set serves both checkpoint selection and fusion fitting, and only 50 Fear examples train the fusion classifier. A stronger future protocol would separate checkpoint selection from fusion fitting or use out-of-fold branch predictions, preserving a final untouched evaluation set. Further work should assess moderate class-weight strengths, class-balanced base-model losses, calibration, conversational context, and repeated-seed performance using development data. These are proposed experiments, not improvements demonstrated here. The existing standalone validation and Trainer test inference paths also differ in precision and batching; unifying and recording these settings should be treated as a new controlled run rather than silently changing the original result table.

### Reproducibility and evidence scope

The result-generation script checks unique ID coverage, target alignment, valid normalized probabilities, argmax consistency, and agreement with the saved JSON metrics. An independent calculation from confusion matrices confirms per-class and aggregate metrics. Source hashes retain the connection between the tables and their underlying predictions. The isolated audit and complete rerun are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md); execution success and numerical agreement are reported separately. A complete rerun of both five-epoch baseline trainings and all fusion stages finished successfully. MacBERT weights and predictions matched exactly, whereas WavLM selected a different checkpoint and changed downstream fusion results; the weighted-fusion alpha changed from 0.75 to 1.00. Separately, standalone replay of the original MacBERT test checkpoint differed on eight labels. Thus, successful end-to-end execution does not establish numerical determinism. The original results remain the basis of this manuscript even when a replay differs.

## Repository evidence

- Data construction: `src/utils/build_manifest.py`, `link_audio_manifest.py`, `mark_audio_valid.py`, and `data/processed/manifest_multimodal.csv`.
- Model methods: `src/text/train_text.py`, `src/speech/train_speech.py`, `configs/*.json`, and the saved best-model configurations/training arguments.
- Fusion methods: `src/fusion/search_weighted_fusion.py`, `evaluate_weighted_fusion.py`, and the two `train_learned_fusion*.py` scripts.
- Reported outcomes: five original per-utterance test prediction CSVs and metrics JSONs, enumerated and hashed in `results/summary/provenance.json`.

This is a repository-grounded experimental draft, not a literature review. Formal bibliographic entries for M3ED, MacBERT, and WavLM should be added and verified for the eventual submission format; no external benchmark or state-of-the-art claim is made here.
