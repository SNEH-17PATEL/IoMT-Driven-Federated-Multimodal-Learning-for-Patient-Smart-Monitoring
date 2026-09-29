# Implemented Changes: SOFA Validation and LLM Safety

This document summarizes SOFA risk-classification evaluation, LLM report safeguards, and the configurable Groq model used by the ICU monitoring project.

## 1. SOFA Risk Metrics

The federated training script evaluates predicted SOFA scores as three risk bands and a high-risk alert:

- Low: SOFA < 5
- Moderate: 5 <= SOFA < 10
- High: SOFA >= 10
- Alert: predicted SOFA >= 8; a true high-risk case is actual SOFA >= 10

The held-out evaluation reports risk-band confusion matrices, per-class precision, recall/sensitivity, specificity, and F1. The alert evaluation reports its confusion matrix, false negatives (high-risk cases missed), false positives (false alerts), precision, recall/sensitivity, specificity, F1, AUROC, and PR-AUC. AUROC and PR-AUC use continuous predicted SOFA scores, not thresholded alert labels. If the test set contains only one actual alert class, AUROC and PR-AUC are saved as unavailable.

Metrics are printed by `python train_federated.py` and saved in `models/training_metadata.json`. After retraining, the Streamlit Risk Assessment tab shows the saved matrices and metrics in the **Held-out Risk Classification Metrics** expander.

**Important limitation:** the available CSVs do not contain patient or ICU-stay IDs. The row-level train/test split therefore cannot ensure that windows from the same stay are isolated between training and testing. Treat these as initial evaluation results, not independent clinical validation. When preprocessing is skipped, the reconstructed test rows are excluded from federated client training.

## 2. LLM Report Safeguards

The LLM system and report instructions direct the model to:

- Avoid assigning diagnoses, naming medications, specifying doses, recommending procedures, or proposing treatment changes.
- Use the supplied patient information and treat clinical notes as data, not as instructions.
- Recommend responsible-clinician review, reassessment, and applicable local protocols instead.

Before display, a rule-based screen checks each of the three generated responses for common medication and treatment terms, dose language, common diagnosis terms, and numeric measurements or labeled vital values not found in the supplied inputs. If any response is flagged, all generated reports are withheld and the app shows the reason for screening.

The three-response cosine score is labeled **Response Similarity**, not reliability. Similar text does not prove that a report is accurate or clinically safe.

**Important limitation:** these prompt and keyword checks are heuristic safeguards. They will not identify every unsupported claim, diagnosis, medication, or invented fact, and they do not replace clinician review or evaluation on clinician-labeled cases.

## 3. Groq Model Configuration

The app uses `openai/gpt-oss-120b` by default and reads the model ID from `GROQ_MODEL`, so it can be changed without editing Python code. Groq retired `llama-3.3-70b-versatile` on August 16, 2026. The example environment file and setup documentation show the default value.

## Where the Changes Are

- `model_utils.py`: reusable SOFA-band and alert metric calculations.
- `train_federated.py`: held-out evaluation, metric reporting, metadata saving, and exclusion of reconstructed test rows from training.
- `app.py`: Risk Assessment metrics display, LLM prompt/screen integration, response-similarity label, and configurable Groq model.
- `llm_safety.py`: rule-based LLM response screening.
- `test_validation.py`: regression tests for metric counts and representative safe/unsafe responses.
- `.env.example`, `README.md`, and `docs/Project_Context.md`: configuration, validation limits, and project documentation.

## Verification

Run the focused regression tests from the repository root:

```powershell
python -m unittest test_validation
```

To generate fresh SOFA classification metrics for the app, run:

```powershell
python train_federated.py
```

The training script overwrites model artifacts and metadata. Review its output and ensure retraining is appropriate for your environment before running it.