# Experiment Log

This file preserves the original five-run experiment record. For verified tables, per-emotion Recall/F1, and confusion matrices, see [results/summary/RESULTS.md](results/summary/RESULTS.md). Methods and interpretation are in [docs/PAPER.md](docs/PAPER.md); execution and numerical reproducibility are tracked separately in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

All reported scores below use the same 4,198 test utterances, seed 42, and seven classes. UAR and Macro-Recall denote the same metric. Differences are descriptive; no statistical significance test was performed.

## Dataset

Dataset: M3ED

Total annotated utterances: 24,449

Empty audio files: 12

Final paired multimodal samples: 24,437

Split:
- Train: 17,420
- Validation: 2,819
- Test: 4,198

Labels:
- Anger
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise


## Experiment 1 — MacBERT Text Baseline

Model:
hfl/chinese-macbert-base

Input:
Text only

Seed:
42

Max length:
32

Test results:
- Accuracy: 0.4507
- Macro-F1: 0.3043
- Macro-Precision: 0.3350
- Macro-Recall: 0.2913

Notes:
- Better balanced performance across classes than WavLM.
- Fear remains difficult due to severe class imbalance.

Result directory:
results/text_macbert_base/


## Experiment 2 — WavLM Speech Baseline

Model:
microsoft/wavlm-base-plus

Input:
Speech only

Sampling rate:
16 kHz

Seed:
42

Test results:
- Accuracy: 0.5012
- Macro-F1: 0.1901
- Macro-Precision: 0.2100
- UAR: 0.2166

Notes:
- Strong bias toward Neutral.
- Good performance on Anger.
- Disgust, Fear, and Surprise were not recognized in the test set.

Result directory:
results/speech_wavlm_base_plus/


## Experiment 3 — Weighted Late Fusion

Method:
Weighted probability-level late fusion

Formula:
P_fusion = 0.75 * P_text + 0.25 * P_speech

Alpha selection:
Selected using validation Macro-F1.

Best validation result:
- Text alpha: 0.75
- Speech alpha: 0.25
- Accuracy: 0.4679
- Macro-F1: 0.3120
- UAR: 0.3087

Test results:
- Accuracy: 0.4740
- Macro-F1: 0.3091
- Macro-Precision: 0.3586
- UAR: 0.2933

Notes:
- Slightly outperformed MacBERT in Macro-F1 and UAR.
- Improved Anger and Neutral classification.
- A single global fusion weight did not improve all emotion classes.
- Learned fusion was subsequently evaluated in Experiments 4 and 5.

Result directory:
results/weighted_fusion/

## Experiment 4 — Learned Fusion (Logistic Regression)

Method:
Probability-level learned fusion using logistic regression.

Input features:
- 7 MacBERT class probabilities
- 7 WavLM class probabilities
- Total: 14 features

Fusion training data:
M3ED validation set (2,819 samples)

Test results:
- Accuracy: 0.5326
- Macro-F1: 0.3038
- Macro-Precision: 0.3543
- UAR: 0.2991

Notes:
- Achieved the highest test accuracy among current methods.
- Achieved higher UAR than weighted late fusion.
- Macro-F1 was slightly lower than weighted late fusion.
- Anger and Neutral benefited strongly from learned fusion.
- Disgust and Fear were not recognized.
- Class imbalance remains a major limitation.

Result directory:
results/learned_fusion_logreg/

## Experiment 5 — Class-Balanced Learned Fusion

Method:
Probability-level learned fusion using logistic regression
with class_weight="balanced".

Input features:
- 7 MacBERT probabilities
- 7 WavLM probabilities
- Total: 14 features

Fusion training data:
M3ED validation set (2,819 samples)

Test results:
- Accuracy: 0.4140
- Macro-F1: 0.3414
- Macro-Precision: 0.3519
- UAR: 0.3628

Notes:
- Achieved the highest Macro-F1 and UAR among all current methods.
- Class balancing substantially improved recall for minority emotions.
- Disgust recall increased to 0.38 and Fear became detectable.
- Happy, Sad, and Surprise recall also improved.
- Neutral recall decreased, resulting in lower overall accuracy.
- The result demonstrates the trade-off between overall accuracy and balanced class performance.

Result directory:
results/learned_fusion_logreg_balanced/

## Minority-class qualification

Balanced fusion correctly recognizes Disgust in 82/218 test cases and Fear in 5/65. Disgust precision is 0.1144 (635 false positives), and Fear precision is 0.0676 (69 false positives). Compared with unweighted LR, Neutral recall falls from 0.7911 to 0.4220. These observations support an improved recall/accuracy trade-off, not a claim that minority-class recognition is solved. Fear has only 280 train and 50 fusion-training examples; Disgust has 1,145 train and 134 fusion-training examples.
