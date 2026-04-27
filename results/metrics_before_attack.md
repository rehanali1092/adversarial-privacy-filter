# Evaluation Metrics — Before Adversarial Attack

## Model: FaceNet (InceptionResnetV1) + Custom Classifier
## Dataset: LFW — 62 classes — 558 test samples

## Results

| Metric | Value |
|---|---|
| Accuracy (Top-1) | 59.50% |
| Accuracy (Top-5) | 65.41% |
| Precision (macro) | 90.17% |
| Recall (macro) | 59.50% |
| F1 Score (macro) | 67.48% |
| Random Baseline | 1.6% |
| Model vs Random | 37x better |

## Interpretation

- **High Precision (90%)**: When model predicts 
  an identity it is correct 90% of the time
- **Moderate Recall (59%)**: Model misses some 
  faces but rarely makes wrong predictions
- **62 classes**: Much harder than binary classification
- **These numbers will DROP after adversarial attack**
- **That drop is the core contribution of this project**

## After Attack (Expected)

| Metric | Before | After FGSM | After PGD |
|---|---|---|---|
| Accuracy | 59.50% | ~15-20% | ~10-15% |
| F1 Score | 67.48% | ~15% | ~10% |

*(To be updated on Day 2)*
