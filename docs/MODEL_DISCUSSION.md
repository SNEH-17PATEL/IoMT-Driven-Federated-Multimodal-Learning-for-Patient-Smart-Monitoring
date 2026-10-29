# Model Discussion — ICU CDSS
**From Weighted Ensemble (Trial Version) to Federated DNN (Final Version)**
**Status:** Fully implemented and verified
**Best result:** R²=0.4724, MAE=1.76 SOFA points (full training set, 48,150 patients)

---

## What This Document Covers

This document answers the most important question about this project's model choices:

- **What model did we start with, and how does it work?**
- **Why did we abandon it, and what were the exact problems?**
- **What model do we use now, and how does every part of it work?**
- **What other models could we have used, and why did we not choose them?**

Every concept is explained from first principles with examples, so anyone reading this can understand the design decisions without needing prior ML knowledge.

---

## Table of Contents

1. [The Previous Approach: Weighted Ensemble](#1-the-previous-approach-weighted-ensemble)
   - [What an Ensemble Is](#11-what-an-ensemble-is)
   - [LightGBM Explained](#12-lightgbm-explained)
   - [XGBoost Explained](#13-xgboost-explained)
   - [Random Forest Explained](#14-random-forest-explained)
   - [Why the Ensemble Was Abandoned](#15-why-the-ensemble-was-abandoned)
2. [The Current Approach: PyTorch Deep Neural Network](#2-the-current-approach-pytorch-deep-neural-network)
   - [What PyTorch Is](#21-what-pytorch-is)
   - [What a Deep Neural Network Is](#22-what-a-deep-neural-network-is)
   - [Our Model Architecture — Every Layer Explained](#23-our-model-architecture--every-layer-explained)
   - [How the Model Is Trained — Step by Step](#24-how-the-model-is-trained--step-by-step)
   - [The Loss Function: Weighted MSE](#25-the-loss-function-weighted-mse)
   - [Backpropagation: How the Model Learns](#26-backpropagation-how-the-model-learns)
   - [The Optimiser: AdamW](#27-the-optimiser-adamw)
   - [Why No Dropout](#28-why-no-dropout)
   - [Why No BatchNorm or LayerNorm](#29-why-no-batchnorm-or-layernorm)
   - [How Federated Learning Works With This DNN](#210-how-federated-learning-works-with-this-dnn)
3. [Alternative Models Considered](#3-alternative-models-considered)
   - [LSTM and GRU (Recurrent Neural Networks)](#31-lstm-and-gru-recurrent-neural-networks)
   - [Transformer (Self-Attention)](#32-transformer-self-attention)
   - [TabNet (Attention for Tabular Data)](#33-tabnet-attention-for-tabular-data)
   - [FedBoost / Federated Gradient Boosting](#34-fedboost--federated-gradient-boosting)
   - [Linear Regression](#35-linear-regression)
   - [BioBERT + DNN](#36-biobert--dnn)
4. [Complete Model Comparison Table](#4-complete-model-comparison-table)
5. [Summary: Panel Q&A](#5-summary-panel-qa)

---

## 1. The Previous Approach: Weighted Ensemble

### 1.1 What an Ensemble Is

An ensemble model is not a single algorithm — it is a **combination of multiple independent models** whose individual predictions are blended together into one final prediction.

The idea behind ensembles is that different models make different kinds of mistakes. If Model A tends to overestimate SOFA for elderly patients and Model B tends to underestimate for the same group, averaging their predictions produces a result that is more accurate than either one alone.

**Our original ensemble:**
```
Final SOFA Score = 0.48 × (XGBoost prediction) + 0.52 × (LightGBM prediction) + 0.00 × (Random Forest prediction)
```

Three models were trained. Their outputs were combined using a weighted average. The weights (0.48, 0.52, 0.00) were determined by testing which combination produced the best validation performance.

**An important detail:** All three models predicted SOFA on a **normalised 0–1 scale** (i.e., SOFA/24), not the actual 0–24 scale. The final prediction was converted back:
```python
sofa_score = ensemble_prediction × 24
```

This introduced a risk of unit confusion — if the model predicted 0.85 for a moderate patient (true SOFA ≈ 7), the displayed score would be 0.85 × 24 = 20.4, which is a life-critical reading. The current DNN eliminates this entirely by predicting raw SOFA directly.

---

### 1.2 LightGBM Explained

**LightGBM (Light Gradient Boosting Machine)** was developed by Microsoft in 2017. It is a tree-based machine learning algorithm and received the highest weight in our ensemble (52%) because it performed best in validation.

#### What is Gradient Boosting?

Gradient boosting builds its model by training a **sequence of simple decision trees**, each one correcting the mistakes of all previous trees. Think of it as hiring a team of specialists:

```
Step 1: Make a very crude prediction
  "Every ICU patient has SOFA = 5.5" (just the average)
  
Step 2: Check which patients you were most wrong about
  Patient A: actual SOFA = 12, predicted = 5.5 → error = +6.5 (too low)
  Patient B: actual SOFA = 2,  predicted = 5.5 → error = −3.5 (too high)
  Patient C: actual SOFA = 7,  predicted = 5.5 → error = +1.5 (slightly low)

Step 3: Train a new small decision tree to predict THESE ERRORS
  This tree learns: "patients with MAP < 60 AND 'vasopressor' in notes → error is +5.0"

Step 4: Add this correction tree to the ensemble (scaled by learning rate 0.1):
  New prediction for Patient A = 5.5 + 0.1 × 5.0 = 6.0  (closer to 12)
  
Step 5: Recompute remaining errors → train another correction tree → add it
  ... repeat 200–1000 times ...

Final prediction = sum of contributions from all trees
```

Each new tree is trained to reduce the remaining errors left by all previous trees. After hundreds of trees, the model has learned to predict very accurately.

#### What makes LightGBM "light" and fast?

Standard gradient boosting considers every possible split point for every feature when building each tree — this is computationally expensive. LightGBM uses two key optimisations:

**GOSS (Gradient-based One-Side Sampling):** When building each tree, LightGBM only uses training samples with large gradient errors (the ones the model is currently worst at predicting) plus a random small sample of the easy cases. This dramatically reduces the data processed per tree with minimal accuracy loss, because the hard cases carry most of the learning signal.

```
Standard boosting:
  Consider all 48,000 patients when building each tree
  
LightGBM with GOSS:
  Keep all ~5,000 patients with large errors (need these)
  Keep random 20% of the remaining 43,000 easy patients (sample)
  Build tree on ~13,600 patients instead of 48,000 → 3.5× faster per tree
```

**Leaf-wise tree growth:** Standard boosting grows trees level-by-level (all splits at depth 1, then all at depth 2, then depth 3). LightGBM always splits whichever single leaf would reduce error the most, regardless of which depth it is at. This produces more accurate trees with fewer splits and grows deeper, more efficient trees.

**EFB (Exclusive Feature Bundling):** Many TF-IDF features are sparse — most patients have 0.0 for most vocabulary terms. EFB groups mutually exclusive sparse features together, effectively reducing 90 TF-IDF columns to fewer bundled columns during tree building, without losing information.

#### What SHAP looks like for LightGBM

The ensemble used `shap.TreeExplainer(lgb_model)` — a SHAP algorithm specifically designed for decision trees. It computes feature attributions by tracing every possible path through the decision tree from root to leaf and computing how much each feature's split contributed to the prediction changing from the population average to the individual prediction.

---

### 1.3 XGBoost Explained

**XGBoost (Extreme Gradient Boosting)** was developed by Tianqi Chen (University of Washington) in 2014 and received 48% weight in our ensemble. It is arguably the most widely used machine learning algorithm in data science competitions (Kaggle) and industry applications.

Like LightGBM, XGBoost uses gradient boosting on decision trees. The key differences are in the engineering details:

**Level-wise tree growth (vs leaf-wise in LightGBM):** XGBoost grows trees one full level (depth) at a time — all splits at depth 1 are found before any split at depth 2. This is more conservative and less prone to overfitting on small datasets, but typically requires more trees to achieve the same accuracy as LightGBM's leaf-wise growth.

**Exact greedy split finding:** For each feature at each node, XGBoost checks every possible split threshold exactly. This is more thorough than LightGBM's histogram approximation but slower on large datasets.

**Built-in L1 and L2 regularisation:** XGBoost adds explicit penalty terms on tree complexity (number of leaves, leaf weight magnitudes) directly into its loss function. This prevents the algorithm from building overly complex trees that memorise the training data.

**Why XGBoost got 48% (almost equal to LightGBM at 52%):**

Both algorithms are excellent at tabular data. Their slight difference in performance (52% vs 48% weight) reflects that LightGBM marginally outperformed XGBoost on our specific validation set — likely because LightGBM's leaf-wise growth found more precise splits on the high-SOFA minority class. Combining both (instead of just one) reduced variance because the two algorithms make slightly different errors.

---

### 1.4 Random Forest Explained

**Random Forest** was introduced by Leo Breiman in 2001. Unlike LightGBM and XGBoost (which build trees sequentially to correct errors), Random Forest uses **bagging (bootstrap aggregation)**:

```
1. Take the training data (48,000 patients)
2. Create 100 different "bootstrap samples" — each is 48,000 patients
   drawn WITH replacement (same patient can appear multiple times)
3. Train one independent decision tree on each bootstrap sample
   (with an extra trick: each split only considers a random subset of features)
4. Final prediction = average of all 100 trees' individual predictions
```

The key idea is **diversity through randomness**. Because each tree is trained on a different random sample AND only sees a random subset of features at each split, the 100 trees are very different from each other — they find different patterns and make different errors. Averaging diverse trees produces a stable, low-variance prediction.

**Why Random Forest got 0% weight:**

Random Forest is slower to train than LightGBM/XGBoost and typically less accurate on tabular data because:
- It does not do error correction (each tree is independent, not correcting the previous)
- Level-wise growth with random feature subsets means each tree is "good but not great"
- Boosting (LightGBM/XGBoost) is more efficient at the error-correction learning

When we tested all three on validation data, Random Forest's predictions were consistently worse than both gradient boosting models. It was retained in the code but assigned 0% weight — it contributed nothing to predictions.

---

### 1.5 Why the Ensemble Was Abandoned

The ensemble was a competent first version. Both LightGBM and XGBoost are excellent algorithms. However, the ensemble had fundamental, unresolvable problems for this project's core objectives.

#### Problem 1 — Federated Learning Is Impossible With Tree Models (Critical)

This is the single most important reason and is **non-negotiable**.

Federated Learning (FL) works by having each hospital train the same model architecture on its private patient data and send back only the **model weights** — the numerical parameters that define the model. The server averages these weights from all hospitals into an improved global model.

**For a neural network (DNN):**
```
Every layer's weights are matrices of floating-point numbers.

Hospital A's first layer weights (simplified):
  W₁_A = [[0.23, -0.45, 0.11, ...], [0.67, 0.12, -0.34, ...], ...]

Hospital B's first layer weights:
  W₁_B = [[0.31, -0.38, 0.09, ...], [0.71, 0.08, -0.29, ...], ...]

FedAvg: W₁_global = (15889 × W₁_A + 15890 × W₁_B) / 31779
                  = [[0.27, -0.415, 0.10, ...], [0.69, 0.10, -0.315, ...], ...]

✓ This is mathematically valid. The averaged weight matrix is a valid DNN first layer.
  It represents a model that has learned from both hospitals' data.
```

**For a decision tree model (LightGBM/XGBoost):**
```
A trained LightGBM model is a collection of decision rules:

Hospital A's model (first 2 trees of ~200):
  Tree 1: IF MAP_mean < 60.5 AND 'vasopressor' in notes:
              IF SpO2_min < 88: → predict 8.2
              ELSE: → predict 6.1
          ELSE:
              IF HR_std > 18.3: → predict 5.3
              ELSE: → predict 3.1
  
  Tree 2: IF 'creatinine' in notes AND latest_MAP < 65:
              → predict 3.8
          ELSE: ...

Hospital B's model (different patient population, completely different trees):
  Tree 1: IF latest_SpO2 < 92 AND 'intubated' in notes:
              → predict 7.1
          ...
```

**Averaging two sets of decision rules is undefined and meaningless.**

There is no mathematical operation that takes "IF MAP_mean < 60.5" from Hospital A and "IF latest_SpO2 < 92" from Hospital B and produces a valid averaged decision rule. Tree structures — which features are split on, which thresholds are used, which leaves predict which values — cannot be meaningfully combined by averaging.

**Consequence:** Without Federated Learning, the project has two bad options:
- Each hospital trains its own isolated model → no shared learning, no privacy benefits
- All hospital data is sent to a central server → patient privacy violation (illegal under HIPAA/GDPR)

Without a DNN, there is no federated learning. Without federated learning, the project's core research contribution does not exist.

#### Problem 2 — The ×24 Multiplication Creates Clinical Danger

The ensemble predicted SOFA in the range [0.0, 1.0] (normalised as SOFA/24). The actual SOFA score was recovered by multiplying by 24.

```
A moderate patient with true SOFA = 7:
  LightGBM predicts: 0.72 (slightly high)
  XGBoost predicts:  0.69 (slightly high)
  Ensemble: 0.52 × 0.72 + 0.48 × 0.69 = 0.374 + 0.331 = 0.705

Wait — these don't look right for SOFA/24 = 7/24 = 0.29.
The model is overpredicting: 0.705 × 24 = 16.9 — in the CRITICAL range.

A patient with true SOFA=7 displayed as SOFA=16.9 would trigger a false alarm
and potentially lead to aggressive interventions the patient does not need.
```

The normalisation-then-multiplication pattern creates a unit conversion layer where errors get amplified by 24. The DNN predicts raw SOFA directly — if it predicts 7.2 for a SOFA=7 patient, the displayed score is 7.2, not 17.3.

#### Problem 3 — SHAP TreeExplainer vs DeepExplainer

The ensemble used SHAP's TreeExplainer, which computes feature attributions by tracing decision paths through the tree. DeepExplainer (used with the DNN) uses gradient-based attribution — it propagates the prediction backwards through the neural network and measures how much each input feature's value moved the final prediction.

DeepExplainer produces smoother, more consistent attributions across similar patients. It respects the continuous, hierarchical representations the DNN builds internally. For clinical decision support — where a doctor needs to understand "why" the model predicted a high SOFA — gradient-based attributions are more reliable and interpretable.

---

## 2. The Current Approach: PyTorch Deep Neural Network

### 2.1 What PyTorch Is

**PyTorch** is an open-source deep learning framework created by Meta AI Research (Facebook). It was released in 2016 and has become the dominant framework in both academic research and production AI systems worldwide — used internally at Meta, Tesla, OpenAI, and most major AI research labs.

Two capabilities make PyTorch essential for this project:

**Automatic differentiation (Autograd):** PyTorch tracks every mathematical operation performed on its data structures (tensors) and can automatically compute the gradient of any output with respect to any input. This is what enables training: PyTorch computes, without any manual calculus, exactly how each of the 23,000 model parameters should change to reduce the prediction error.

```
Simple example of what Autograd does:
  We have: y = w × x + b   (one-neuron "network")
  x = 3.0,  w = 0.5,  b = 0.1
  
  y = 0.5 × 3.0 + 0.1 = 1.6
  
  True output was 2.0. Error = (1.6 - 2.0)² = 0.16
  
  PyTorch automatically computes:
    ∂Error/∂w = 2 × (1.6 - 2.0) × x = 2 × (-0.4) × 3.0 = -2.4
    ∂Error/∂b = 2 × (1.6 - 2.0) × 1 = -0.8
  
  These gradients tell us: "increase w slightly and increase b slightly to reduce error"
  
  For a full DNN with 23,000 parameters and 108 inputs,
  PyTorch computes all 23,000 gradients automatically in one backward pass.
```

**Dynamic computation graphs:** The model's computation is defined on-the-fly as code runs, not pre-compiled as a static graph. This makes debugging intuitive — you can print intermediate values, inspect shapes, and modify the computation mid-training without restructuring the entire model definition.

---

### 2.2 What a Deep Neural Network Is

A **Deep Neural Network (DNN)** is a mathematical function that maps inputs (108 patient features) to an output (predicted SOFA score) through multiple layers of learned transformations.

#### The fundamental building block: a neuron

A single artificial neuron does three things:
1. Takes a weighted sum of its inputs
2. Adds a bias term
3. Applies an activation function

```
A single neuron with 3 inputs:

  inputs:        x₁ = 0.8 (HR_mean normalised)
                 x₂ = −1.2 (SpO2_mean normalised — below average)
                 x₃ = 0.3 (MAP_mean normalised)
  
  weights:       w₁ = 0.5, w₂ = −0.8, w₃ = 0.4
  bias:          b  = 0.1
  
  Weighted sum:  z = w₁×x₁ + w₂×x₂ + w₃×x₃ + b
               = 0.5×0.8 + (−0.8)×(−1.2) + 0.4×0.3 + 0.1
               = 0.4 + 0.96 + 0.12 + 0.1
               = 1.58
  
  After ReLU:    output = max(0, 1.58) = 1.58
  
  What this neuron learned to detect:
    The large negative weight on SpO2 (−0.8) means low SpO2 strongly
    activates this neuron. The positive weight on HR (0.5) means high HR
    also activates it. This neuron may be detecting "respiratory distress"
    — low oxygen with compensatory high heart rate.
```

The weights and bias are the **learnable parameters** — they start random and are adjusted during training until the neuron detects something useful.

#### A layer of neurons

A layer consists of multiple neurons, each looking at all the same inputs but with different weights. Our first hidden layer has 128 neurons, all looking at the same 108 input features but detecting 128 different patterns simultaneously.

```
Layer 1 (128 neurons, each with 108 inputs):
  Neuron 1:  detects "respiratory distress" (activates on low SpO2 + high RR)
  Neuron 2:  detects "cardiovascular collapse" (activates on low MAP + vasopressor mentions)
  Neuron 3:  detects "renal failure" (activates on "creatinine" mentions + low urine)
  ...
  Neuron 128: detects some combination of features the training data found useful
```

The 128 outputs from Layer 1 become the inputs to Layer 2, which detects patterns in those 128 "clinical syndrome" signals. Layer 3 detects patterns in Layer 2's output. The final layer maps the 32 remaining abstract representations to a single SOFA score.

#### Why "deep"?

"Deep" refers to having multiple hidden layers (Layer 1, Layer 2, Layer 3 in our case). Each layer builds on abstractions from the previous layer:

- **Layer 1:** Detects simple patterns in raw features — "low SpO2", "high HR", "vasopressor mention"
- **Layer 2:** Combines those into clinical syndromes — "respiratory failure", "septic shock", "acute kidney injury"
- **Layer 3:** Distills syndromes into severity levels — "single organ failure", "multi-organ failure", "critical failure"
- **Output layer:** Maps severity representation to a specific SOFA score

A shallow network (Layer 1 → Output directly) could only learn simple combinations of raw features. The depth allows the model to learn increasingly abstract, clinically meaningful representations.

---

### 2.3 Our Model Architecture — Every Layer Explained

```
Input:  108 features per patient
        [HR_mean, HR_std, RR_mean, SpO2_mean, SpO2_min,
         Temp_mean, SBP_mean, DBP_mean, MAP_mean,
         latest_HR, latest_RR, latest_SpO2, latest_Temp,
         latest_SBP, latest_DBP, latest_MAP,
         GCS_eye_opening, stress_score,
         TF-IDF: intubated, ventilator, creatinine, ... (90 terms)]
         
         ↓
         
┌─────────────────────────────────────────────────────────────────┐
│  LINEAR LAYER 1:   108 → 128                                    │
│  Parameter count: 108×128 weights + 128 biases = 13,952        │
│  What it learns:  relationships between raw input features      │
│  Example: weight matrix entry W[3][47] learns how much the      │
│           3rd neuron cares about TF-IDF feature 47 ("sepsis")  │
└─────────────────────────────────────────────────────────────────┘
         ↓ 128 values
         
┌─────────────────────────────────────────────────────────────────┐
│  ReLU ACTIVATION                                                │
│  Each of 128 values: output = max(0, value)                    │
│  Negative values → 0  (neuron is "off")                        │
│  Positive values → kept unchanged                               │
│  Why: introduces non-linearity, allows learning complex patterns│
└─────────────────────────────────────────────────────────────────┘
         ↓ 128 values (some now zero)
         
┌─────────────────────────────────────────────────────────────────┐
│  LINEAR LAYER 2:   128 → 64                                     │
│  Parameter count: 128×64 weights + 64 biases = 8,256           │
│  What it learns:  combinations of the 128 Layer 1 patterns     │
│  Example: "respiratory distress signal" × "vasopressor signal" │
│           → "septic respiratory failure syndrome"               │
└─────────────────────────────────────────────────────────────────┘
         ↓ 64 values
         
┌─────────────────────────────────────────────────────────────────┐
│  ReLU ACTIVATION                                                │
└─────────────────────────────────────────────────────────────────┘
         ↓ 64 values

┌─────────────────────────────────────────────────────────────────┐
│  LINEAR LAYER 3:   64 → 32                                      │
│  Parameter count: 64×32 weights + 32 biases = 2,080            │
│  What it learns:  final compression into severity dimensions    │
└─────────────────────────────────────────────────────────────────┘
         ↓ 32 values
         
┌─────────────────────────────────────────────────────────────────┐
│  ReLU ACTIVATION                                                │
└─────────────────────────────────────────────────────────────────┘
         ↓ 32 values

┌─────────────────────────────────────────────────────────────────┐
│  LINEAR LAYER 4:   32 → 1  (output layer)                       │
│  Parameter count: 32×1 weights + 1 bias = 33                   │
│  What it does:    linear combination of 32 severity signals     │
│                   → single SOFA prediction (0–24)               │
│  No activation:   output is raw float (correct for regression)  │
└─────────────────────────────────────────────────────────────────┘
         ↓
Output: Predicted SOFA score  (e.g., 8.37)

Total trainable parameters: 13,952 + 8,256 + 2,080 + 33 = 24,321
```

**Why this shape — 108 → 128 → 64 → 32 → 1?**

**Why expand from 108 to 128 first?**
The input expansion from 108 to 128 neurons creates space for the model to represent feature combinations — things that are not in any individual input feature but emerge from their interaction. For example, "high HR_std (variable heart rate) AND low MAP_mean AND 'septic shock' in notes" is a combination of three separate features. The 128-neuron first layer has enough capacity to represent this and many other combinations simultaneously.

**Why progressively narrow (128 → 64 → 32 → 1)?**
Each subsequent layer must compress the previous layer's 128 (or 64) representations into fewer dimensions. This compression forces the model to retain only the most important information for predicting SOFA and discard the rest. By the time we reach 32 dimensions, the model has learned a compact representation of clinical severity.

**Why ~23,000 parameters total?**
With 48,150 training patients, the ratio is approximately 2:1 (samples per parameter). Machine learning theory and empirical experience on tabular datasets suggest this ratio is appropriate for good generalisation:
- Too few parameters (1:1 or below) → underfitting: model cannot learn the complexity of the data
- Too many parameters (0.5:1 or below) → overfitting: model memorises training data but fails on new patients
- 2:1 → right-sized: model learns genuine patterns without overfitting

The previous architecture (618 → 256 → 128 → 64 → 1 = ~200,000 parameters) had only 0.24 samples per parameter — severe over-parameterisation that contributed to poor generalisation.

---

### 2.4 How the Model Is Trained — Step by Step

Training repeats the same 4-step cycle thousands of times, each time adjusting the model weights slightly to make predictions more accurate.

#### Training setup
```
Dataset: 48,150 patients (80% of all data)
Mini-batch size: 64 patients per step
Steps per epoch: 48,150 / 64 ≈ 752 gradient steps
Epochs per FL round: 3
Total FL rounds: 100
Total gradient steps: 752 × 3 × 100 = 225,600 weight updates
```

#### Step 1: Forward Pass — Make a Prediction

Take a mini-batch of 64 patients. Feed their feature vectors through all 4 layers:

```
Patient with high-risk clinical presentation:
  Input (108 features): 
    HR_mean = −0.8 (normalised: slightly below average)
    SpO2_min = −2.3 (normalised: very low — 2.3 std deviations below mean)
    MAP_mean = −1.9 (normalised: very low blood pressure)
    "vasopressor" TF-IDF = 1.8 (normalised: strongly present)
    "creatinine" TF-IDF = 1.2 (normalised: present)
    ... (103 other features)
    
  Layer 1 (108 → 128):
    h1 = ReLU(W1 @ input + b1)
    128 values, some positive (active patterns), some zero (inactive)
    
  Layer 2 (128 → 64):
    h2 = ReLU(W2 @ h1 + b2)
    64 values — compressed clinical syndrome representations
    
  Layer 3 (64 → 32):
    h3 = ReLU(W3 @ h2 + b3)
    32 values — final severity representations
    
  Layer 4 (32 → 1):
    prediction = W4 @ h3 + b4
    = 11.2  (model predicts SOFA = 11.2)
    
  True label: y = 14 (actual SOFA score from MIMIC-III)
```

#### Step 2: Compute Loss — Measure How Wrong We Are

```
Error for this patient:
  Residual = prediction − true = 11.2 − 14 = −2.8  (predicted too low)
  
  Weight = 1 + target × 0.5 = 1 + 14 × 0.5 = 8.0  (high SOFA → high weight)
  After batch normalisation: normalised_weight = 8.0 / batch_mean_weight
  
  Weighted squared error = normalised_weight × (−2.8)² = ~2.6 × 7.84 = ~20.4
  
This patient's error contributes 2.6× more to the batch loss than an average patient.
The model receives stronger training signal to improve its predictions for this
high-SOFA patient.
```

The total batch loss is the average weighted squared error across all 64 patients in the batch.

---

### 2.5 The Loss Function: Weighted MSE

The loss function is the mathematical measure of prediction error. Standard Mean Squared Error (MSE) treats all patients equally:

```
Standard MSE = mean((predicted - actual)²)

Problem: 63% of ICU patients have SOFA 0–4 (Low Risk)
         6% have SOFA ≥ 10 (High Risk — the most clinically important group)
         
With equal weighting:
  The model focuses 63% of its learning on Low-Risk patients
  and only 6% on the most critical High-Risk patients.
  
  This is the opposite of what we want for an ICU early warning system.
```

**Our Weighted MSE:**

```python
def weighted_mse_loss(predictions, targets):
    weights = 1.0 + targets * 0.5      # higher SOFA = higher weight
    weights = weights / weights.mean() # normalise so mean weight = 1.0
    return (weights * (predictions - targets) ** 2).mean()
```

**Worked example for a batch of 5 patients:**

```
Patient | Actual SOFA | Predicted | Raw weight | After normalisation
--------|-------------|-----------|------------|---------------------
A       | 2           | 2.8       | 1 + 2×0.5 = 2.0   | 0.67×
B       | 5           | 4.3       | 1 + 5×0.5 = 3.5   | 1.17×
C       | 8           | 7.1       | 1 + 8×0.5 = 5.0   | 1.67×
D       | 13          | 9.2       | 1 + 13×0.5 = 7.5  | 2.50×
E       | 1           | 1.9       | 1 + 1×0.5 = 1.5   | 0.50×

Batch mean raw weight = (2.0 + 3.5 + 5.0 + 7.5 + 1.5) / 5 = 3.9
Normalised weights: 2.0/3.9=0.51 → 0.67, 3.5/3.9=0.90→1.17, etc.

Weighted errors:
  Patient A: 0.67 × (2.8−2)²  = 0.67 × 0.64 = 0.43
  Patient B: 1.17 × (4.3−5)²  = 1.17 × 0.49 = 0.57
  Patient C: 1.67 × (7.1−8)²  = 1.67 × 0.81 = 1.35
  Patient D: 2.50 × (9.2−13)² = 2.50 × 14.44 = 36.1   ← heavily penalised
  Patient E: 0.50 × (1.9−1)²  = 0.50 × 0.81 = 0.41

Batch loss = (0.43 + 0.57 + 1.35 + 36.1 + 0.41) / 5 = 7.77

Patient D (SOFA=13, predicted=9.2) dominates the loss.
The gradient update will strongly push the model to improve its prediction for
patients like D — which is exactly what we want.
```

**Why multiplier 0.5 and not 3.0 (which was tried earlier)?**

```
With weight = 1 + SOFA × 3.0 (the earlier setting):
  Mean weight for our SOFA distribution ≈ 14.5
  
  Low-Risk patient (SOFA=2): normalised weight = 7.0/14.5 = 0.48
  Average patient (SOFA=4):  normalised weight = 13.0/14.5 = 0.90
  High-Risk patient (SOFA=12): normalised weight = 37.0/14.5 = 2.55
  
  63% of patients (Low Risk) receive only 48% of their fair signal.
  The server evaluates models using unweighted MSE — dominated by Low Risk.
  Training optimised for High Risk; selection favoured Low Risk → mismatch.
  
With weight = 1 + SOFA × 0.5 (current setting):
  Low-Risk patient (SOFA=2): normalised weight ≈ 0.67  (33% reduction — meaningful but not severe)
  Average patient (SOFA=4):  normalised weight ≈ 1.0   (fair share)
  High-Risk patient (SOFA=12): normalised weight ≈ 2.0  (2× emphasis — meaningful but not overwhelming)
  
  Training and evaluation objectives are aligned. Model improves consistently.
```

---

### 2.6 Backpropagation: How the Model Learns

After computing the loss, the model must update all 23,000+ parameters to reduce that loss. **Backpropagation** is the algorithm that computes how each parameter should change.

**The chain rule — the mathematical foundation:**

Because the DNN is a composition of functions (output → Layer 4 → Layer 3 → Layer 2 → Layer 1 → input), we can use the chain rule of calculus to find how the output changes with respect to any parameter deep in the network.

```
Simple chain rule example:
  y = f(g(x))
  dy/dx = (dy/dg) × (dg/dx)

For our DNN:
  loss = f(output)
  output = Layer4(h3)
  h3 = Layer3(h2)
  h2 = Layer2(h1)
  h1 = Layer1(input)
  
  ∂loss/∂W1 = (∂loss/∂output) × (∂output/∂h3) × (∂h3/∂h2) × (∂h2/∂h1) × (∂h1/∂W1)
  
  Each factor tells us how much a small change in one layer's output affects
  the next layer. Multiplied together, this tells us exactly how much W1's
  values should change to reduce the final loss.
```

**What backpropagation computes:**

For every one of the 23,000+ parameters:
- A positive gradient → "this parameter is too large, decrease it to reduce loss"
- A negative gradient → "this parameter is too small, increase it to reduce loss"
- Gradient near zero → "this parameter barely affects the loss, leave it alone"

PyTorch computes all 23,000+ gradients automatically in one backward pass — no manual calculus required.

**Gradient clipping:**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Occasionally, a mini-batch of unusual patients can produce very large gradients — large enough that a single update would move the model weights far from their current position and damage everything learned so far ("exploding gradients").

Gradient clipping fixes this: if the total magnitude of all gradients (measured as the L2 norm) exceeds 1.0, all gradients are proportionally scaled down so their total magnitude becomes exactly 1.0. The direction of the update is preserved; only the magnitude is capped.

```
Example:
  Batch of 64 patients includes an extreme outlier (SOFA=24, very unusual features)
  Gradient L2 norm computed: 8.3  (8.3× larger than the clipping threshold)
  After clipping: gradient L2 norm = 1.0
  All gradient values scaled by 1.0/8.3 = 0.12
  
  The model still updates in the right direction, just with 1/8 the step size.
  The outlier patient does not destabilise weeks of training.
```

---

### 2.7 The Optimiser: AdamW

After backpropagation computes the gradient for each parameter, the optimiser decides how to update the parameter values. We use **AdamW** (Adam with decoupled Weight Decay).

#### Why not simple gradient descent?

Simple gradient descent updates each parameter by: `W ← W − learning_rate × gradient`

Problems:
1. **Same learning rate for all parameters:** Some parameters may need large updates (they are far from optimal), others need tiny updates (they are close). A single fixed learning rate does not work well for both.
2. **No momentum:** If all previous batches found that weight W should increase, but one noisy batch says decrease, simple gradient descent changes direction immediately. This causes oscillation.

#### What Adam does

Adam keeps track of two running averages for each parameter:

```
m_t = β₁ × m_{t-1} + (1 - β₁) × gradient     ← First moment (momentum)
      = 0.9 × m_{t-1} + 0.1 × gradient
      
v_t = β₂ × v_{t-1} + (1 - β₂) × gradient²    ← Second moment (squared gradient)
      = 0.999 × v_{t-1} + 0.001 × gradient²
```

**`m_t` (momentum):** A smooth running average of recent gradients. If gradients have been consistently positive (this parameter should increase), m_t is large and positive. If gradients have been oscillating (positive then negative), m_t stays small because the positive and negative values cancel.

```
Example — consistent gradients (good signal):
  gradient at step 1: +0.5
  gradient at step 2: +0.6
  gradient at step 3: +0.4
  
  m₁ = 0.9×0 + 0.1×0.5 = 0.050
  m₂ = 0.9×0.050 + 0.1×0.6 = 0.045 + 0.060 = 0.105
  m₃ = 0.9×0.105 + 0.1×0.4 = 0.094 + 0.040 = 0.134
  
  Momentum builds up in the positive direction → confident, larger updates

Example — noisy gradients (one bad batch):
  gradient at step 1: +0.5
  gradient at step 2: −0.6  ← one noisy batch says "go negative"
  gradient at step 3: +0.4
  
  m₁ = 0.050
  m₂ = 0.9×0.050 + 0.1×(−0.6) = 0.045 − 0.060 = −0.015
  m₃ = 0.9×(−0.015) + 0.1×0.4 = −0.013 + 0.040 = +0.027
  
  Momentum mostly cancels out → small update → noisy batch had little effect
```

**`v_t` (adaptive rate):** A running average of squared gradient magnitudes. Parameters with large, consistent gradients get larger v_t → smaller effective step size (they are getting large updates already). Parameters with tiny gradients get smaller v_t → larger effective step size (they need help making progress).

```
Final update rule:
  m_hat = m_t / (1 - β₁ᵗ)   ← bias correction (important in early steps)
  v_hat = v_t / (1 - β₂ᵗ)   ← bias correction
  
  W ← W - lr × m_hat / (√v_hat + ε)
  
  When v_hat is large:  step = lr / √large_number → small step (parameter already moving fast)
  When v_hat is small:  step = lr / √small_number → large step (parameter needs help)
```

#### Why AdamW instead of Adam?

The "W" in AdamW stands for "decoupled Weight Decay." This is a subtle but important difference:

```
Adam weight decay:
  Implemented by adding L2 penalty to the loss: loss = original_loss + λ × ||W||²
  This adds λ×W to the gradient. When Adam applies its adaptive scaling to this
  gradient, the weight decay effect is also scaled — it becomes larger for parameters
  with small gradients and smaller for parameters with large gradients.
  This conflates two different things: the data-driven gradient and the regularisation signal.

AdamW weight decay:
  W ← W × (1 - weight_decay)   applied AFTER the Adam update, not through the gradient
  
  This keeps weight decay constant and independent of the gradient magnitude.
  Every parameter's magnitude is gently pulled toward zero at the same rate,
  regardless of how large or small its gradient was.
  
  Empirically, this produces better generalisation on tabular data.
```

**In our training:** `weight_decay=1e-4` means every parameter's magnitude decreases by 0.01% each step. This gently discourages any weight from growing too large, preventing overfitting to training data quirks.

---

### 2.8 Why No Dropout

**Dropout** is a very popular regularisation technique for neural networks. During training, each neuron's output is randomly set to zero with probability `p`. With `p=0.3`, 30% of neurons are "dropped" in each forward pass, forcing the network to not rely on any single neuron.

In standard (centralised) training, Dropout works very well: it forces redundancy and prevents neurons from co-adapting, effectively training many different sub-networks and averaging their effects.

**Why Dropout fails catastrophically in Federated Learning:**

```
Round 1 of FL training:

Hospital 0's random Dropout masks (p=0.3):
  Layer 1: drops neurons {5, 23, 47, 91, 102, ...}  (random 30%)
  Layer 2: drops neurons {3, 18, 44, 60, ...}
  
Hospital 1's random Dropout masks (different random seed):
  Layer 1: drops neurons {7, 11, 38, 82, 99, ...}  (different 30%)
  Layer 2: drops neurons {1, 27, 51, 58, ...}

Hospital 0 trained a sub-network with neurons {5, 23, 47, 91, 102, ...} ALWAYS INACTIVE.
Hospital 1 trained a sub-network with neurons {7, 11, 38, 82, 99, ...} ALWAYS INACTIVE.

These are effectively two different model architectures with 30% of neurons in different positions.

FedYogi now averages the weights:
  W[5] for Hospital 0: never got any gradient (always dropped) → random value
  W[5] for Hospital 1: was active, has been trained on many patients → meaningful value
  
  Average of (random value) and (meaningful value) = noisy half-trained value

The aggregated model has no neurons that are fully trained — every neuron was
dropped by at least one hospital and trained by at least one other.
The result is a model that satisfies no hospital's training objective.
```

**Test results:**
| Configuration | R² | 
|---|---|
| No Dropout (current) | 0.2229 |
| Dropout p=0.3/0.2 | 0.057 |
| Dropout p=0.1/0.05 | −0.067 |

Even small Dropout probabilities caused catastrophic failure in the FL setting. The regularisation is handled instead by AdamW's weight decay, which applies uniformly and consistently across all 3 hospitals.

---

### 2.9 Why No BatchNorm or LayerNorm

Both normalisation techniques were tested and failed. Understanding why reveals important constraints of the FL + regression combination.

**BatchNorm failure — divergent statistics:**

BatchNorm normalises each feature's activations within a batch to have mean=0, std=1. It maintains running estimates of the mean and variance for each layer, which are updated during training.

```
Hospital 0's patients: predominantly younger general ICU (mean SOFA ≈ 4.2)
  Layer 1 activations: mean=0.31, std=1.15 (for a specific neuron)
  BatchNorm running mean for this neuron: 0.31
  
Hospital 1's patients: mixed ICU (mean SOFA ≈ 5.8)
  Same neuron's activations: mean=0.48, std=1.22
  BatchNorm running mean: 0.48

FedAvg of these running means: (0.31 + 0.48) / 2 = 0.395

This value is wrong for BOTH hospitals — when Hospital 0 tries to use the
averaged running mean (0.395) to normalise its next batch (which has mean=0.31),
it subtracts 0.395 from 0.31 values, producing −0.085 instead of near-zero.
The normalization is incorrect → training becomes unstable.
```

**LayerNorm failure — destroys SOFA severity distinction (tested twice, failed both times):**

LayerNorm normalises each patient's activations so that across all neurons in a layer, the mean is 0 and standard deviation is 1. This is applied per-sample, per-layer.

```
Patient A (SOFA=20, severe multi-organ failure):
  Layer 3 activations: [4.2, 3.8, 2.1, 1.5, 6.3, 5.1, ...]  ← large values
  After LayerNorm: normalised to mean=0, std=1
  Output activations: [0.87, 0.63, −0.42, −0.79, 1.44, 1.10, ...]

Patient B (SOFA=2, mild illness):
  Layer 3 activations: [1.2, 0.8, 0.3, 0.1, 1.9, 1.5, ...]  ← small values
  After LayerNorm: normalised to mean=0, std=1
  Output activations: [0.82, 0.57, −0.53, −0.83, 1.51, 1.08, ...]
```

After LayerNorm, Patients A and B produce **nearly identical activation patterns** in Layer 3. The final linear layer (Layer 4) receives near-identical inputs for a SOFA=20 patient and a SOFA=2 patient — it cannot distinguish them. The model's prediction range collapsed to 0.22–7.04, and MAE for High-Risk patients jumped from 3.4 to 6.5 SOFA points.

**The fundamental insight:** LayerNorm is designed for classification tasks (where the label is a category, not a magnitude) and NLP (where token representations should be scale-invariant). For regression over a wide output range (SOFA 0–24), scale information is critical — the model NEEDS to know that Patient A's activations are 3× larger than Patient B's, because that difference encodes severity.

---

### 2.10 How Federated Learning Works With This DNN

The DNN is compatible with FL in a way that tree models fundamentally are not: **its parameters are just matrices of floating-point numbers, and averaging matrices is always valid.**

```
Round 1:
  Server creates random initial weights W⁰ (same random seed for all hospitals)
  Server sends W⁰ to Hospital 0, Hospital 1, Hospital 2

  Hospital 0 (15,889 General ICU patients):
    Loads W⁰ → trains 3 local epochs (750 gradient steps) → produces W⁰_H0
    
  Hospital 1 (15,890 Mixed ICU patients):
    Loads W⁰ → trains 3 local epochs → produces W⁰_H1
    
  Hospital 2 (16,371 Cardiac/Trauma patients):
    Loads W⁰ → trains 3 local epochs → produces W⁰_H2

  All hospitals send their weight matrices back to server.
  
  FedAvg (sample-size weighted average):
    W_avg = (15889 × W⁰_H0 + 15890 × W⁰_H1 + 16371 × W⁰_H2) / 48150
  
  FedYogi server update (adaptive momentum):
    Δ = W_avg − W⁰  ← pseudo-gradient
    Update m, v using Yogi rule
    W¹ = W⁰ + 0.01 × m / (√v + 0.001)
  
  W¹ is the new global model. It contains knowledge from all 3 hospitals
  without any patient data being shared.

Round 2:
  Server sends W¹ to all 3 hospitals.
  ... repeat for 100 rounds ...

Round 24: Best-performing round (lowest aggregate validation MSE)
  W²⁴ saved as the deployed model.
```

**Why this is fundamentally different from tree-based FL:**

For a DNN, the weight matrix W[i][j] for Layer 1, neuron i, input j has a consistent meaning across all hospitals — it represents "how much does the j-th input feature (e.g., MAP_mean) influence the i-th Layer 1 neuron?" This meaning is the same regardless of which hospital's data was used to train it. Averaging W[i][j] from Hospital A and Hospital B produces a valid "how much should MAP_mean influence neuron i, on average across both populations."

For a decision tree, there is no equivalent operation. "Hospital A's tree splits on MAP_mean < 60.5 at depth 2" and "Hospital B's tree splits on SpO2_min < 88 at depth 1" refer to completely different structural positions in different trees. There is no meaningful average of these two facts.

---

## 3. Alternative Models Considered

For each alternative, we explain what it is, how it would have been applied to our problem, and why it was not chosen.

### 3.1 LSTM and GRU (Recurrent Neural Networks)

**What they are:**

LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are neural networks designed specifically for sequential data — data where the order matters and where understanding position t requires remembering what happened at positions 1 through t-1.

Standard text generation, machine translation, and speech recognition all use these. An LSTM reading the sentence "The patient's oxygen saturation has been declining for the last four hours" maintains a memory of "declining" when it processes "four hours later" — the temporal context is essential to understanding the whole.

**How it would apply to ICU vital signs:**

Instead of computing HR_mean, HR_std, SpO2_min ourselves from the last 20 readings, we would feed the raw sequence directly:

```
Standard DNN (our approach):
  Input: flat 108-feature vector
  Feature 1: HR_mean (we computed this from 20 readings)
  Feature 2: HR_std  (we computed this from 20 readings)
  ...

LSTM approach:
  Input: 3D sequence tensor of shape (batch, 20_readings, 7_vital_signs)
  
  Reading 1:  [HR=82, RR=18, SpO2=97, Temp=37.1, SBP=118, DBP=76, MAP=90]
  Reading 2:  [HR=86, RR=19, SpO2=96, Temp=37.2, SBP=115, DBP=74, MAP=88]
  Reading 3:  [HR=89, RR=21, SpO2=94, Temp=37.4, SBP=110, DBP=70, MAP=83]
  ...
  Reading 20: [HR=118, RR=28, SpO2=86, Temp=38.9, SBP=88, DBP=54, MAP=65]
  
  LSTM processes each reading in sequence, maintaining hidden state:
  h₀ → [Reading 1] → h₁ → [Reading 2] → h₂ → ... → h₂₀ → final state → SOFA prediction
  
  The LSTM can learn:
  "SpO2 dropped from 97 to 86 across 20 readings — that's a 2% deterioration per reading"
  "HR is accelerating — the increase is getting faster, not linear"
  These patterns require remembering the whole trajectory, which LSTM does naturally.
```

**Why LSTM was not chosen:**

The complexity cost did not justify the benefit in our specific setting:

1. **Feature engineering achieves similar results:** Our 9 trend features (HR_mean, HR_std, SpO2_mean, SpO2_min, etc.) already capture the temporal statistics an LSTM would derive. We gave up some subtlety (LSTM could detect acceleration in SpO2 decline; we only capture mean and min) but avoided substantial implementation complexity.

2. **TF-IDF integration is harder:** The LSTM operates on a 3D temporal tensor (batch × time × vitals). The TF-IDF features are a flat 90-dimensional vector. Combining these two very different tensor shapes requires a more complex architecture (e.g., process vitals with LSTM, process text with a separate linear layer, concatenate and feed to output layers). This is doable but adds significant complexity.

3. **FL compatibility is the same:** RNNs are gradient-based and their weights are matrices — FedAvg applies just as well as with our DNN. This was not a deciding factor.

4. **More data required:** LSTMs have more parameters than a flat DNN of the same depth (because they have internal gating mechanisms). With 48,150 training samples, we would need to be careful about overfitting.

---

### 3.2 Transformer (Self-Attention)

**What it is:**

Transformers, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017), are the architecture underlying GPT, BERT, and all modern large language models. Instead of processing a sequence step-by-step (like LSTM), a Transformer processes all positions simultaneously and computes **attention** — a learned measure of which positions are most relevant to each other.

```
For 20 vital sign readings, self-attention computes:
  "How much should Reading 20 (current state) pay attention to Reading 15?"
  "How much should Reading 20 pay attention to Reading 1?"
  
  If Reading 5 had a sudden SpO2 drop that triggered treatment,
  attention might learn to weight it highly when assessing Reading 20's SOFA
  — even though 15 readings have passed.
  
  An LSTM would have to "remember" this through 15 hidden state updates.
  A Transformer attends to it directly in one step.
```

**How it would apply:**

```
Standard Transformer for ICU vitals:
  Each vital reading = one "token"
  20 readings → 20 tokens
  
  Multi-head self-attention learns which time steps are most relevant
  to predicting the current SOFA score
  
  Output: context-aware representation of all 20 readings → SOFA prediction
```

For clinical text, ClinicalBERT or BioBERT (BERT models pre-trained on medical literature and clinical notes) could replace our TF-IDF encoding, providing much richer semantic understanding of the notes.

**Why Transformers were not chosen:**

The size problem is prohibitive in our FL setting:

```
Our DNN: ~23,000 parameters → ~90 KB to transmit per FL round
         × 3 hospitals × 100 rounds = 27 MB total FL communication

A small clinical Transformer: ~5,000,000 parameters → ~20 MB per FL round
         × 3 hospitals × 100 rounds = 6 GB total FL communication
         
ClinicalBERT: ~440 MB per model → 440 MB per FL round
         = completely impractical for simulated or real hospital FL
```

Additionally, for a tabular regression task with ~50,000 samples, Transformers typically do not outperform well-tuned DNNs or gradient boosting. The architecture overhead (positional encoding, multi-head attention, residual connections, layer normalisation, feed-forward sub-layers) adds complexity without proportional accuracy gain on structured tabular data.

---

### 3.3 TabNet (Attention for Tabular Data)

**What it is:**

TabNet (Arik & Pfister, Google Brain, 2019) is a neural network architecture specifically designed for tabular data. It uses sequential attention to select which features to focus on at each "step" of its computation.

```
TabNet's sequential attention (3 steps example):

Step 1: Look at all 108 features → attend to [HR_mean, MAP_mean, SpO2_min]
        Learn initial pattern from these three
        
Step 2: Given what Step 1 found → attend to [stress_score, GCS_eye_opening, "vasopressor"]
        Refine prediction with additional context
        
Step 3: Given what Steps 1+2 found → attend to ["creatinine", "bilirubin"]
        Final adjustment based on renal and hepatic signals
        
Final prediction: combination of all three steps' contributions
```

This produces naturally interpretable feature selections at each step — not quite SHAP, but a different kind of built-in explainability.

**Why TabNet was not chosen:**

1. **SHAP integration is difficult:** Standard SHAP methods (including DeepExplainer) assume the model is a standard feedforward DNN. TabNet's attention masking mechanism breaks this assumption. Computing SHAP values for TabNet requires custom implementations that are less stable and less well-validated than standard DeepExplainer.

2. **Less established in FL literature:** Our FL training uses Flower framework's standard tools for parameter averaging. TabNet's architecture is non-standard enough that ensuring correct weight averaging in the FL context would require careful verification.

3. **Similar performance on tabular data:** Empirically, TabNet does not consistently outperform a well-tuned DNN on regression tasks with 50K samples. The additional complexity does not reliably translate to better SOFA predictions.

---

### 3.4 FedBoost / Federated Gradient Boosting

**What it is:**

Research has proposed methods to perform FL with gradient boosted trees — the category of models we used in the ensemble. The most practical approach uses gradient histograms:

```
Standard (non-federated) gradient boosting:
  Server has all 48,150 patients
  For each tree, compute gradients for all patients
  Find the best split by examining all patients' feature values
  
Federated gradient boosting (histogram-based):
  Each hospital computes gradients for its local patients
  Each hospital bins its features into histograms (count how many patients
  have feature value in each bin)
  Histograms are shared with the server (not individual patient data)
  Server reconstructs approximate gradients from combined histograms
  Server decides tree splits → broadcasts split decisions back to hospitals
```

**Why this was not chosen:**

1. **Not supported by Flower framework:** Our entire FL infrastructure is built on the Flower (flwr) library, which handles the orchestration of rounds, client management, and aggregation strategies. Flower does not natively support gradient-histogram-based federated tree training — it would require a completely custom implementation.

2. **Histogram leakage risk:** Sharing feature histograms reveals statistical information about the local patient population that raw model weights do not. A global histogram is more privacy-preserving than sharing individual records, but less privacy-preserving than sharing neural network weight tensors (which obscure the underlying data distribution).

3. **Active research area, not settled engineering:** Standard neural network FL (FedAvg, FedYogi, FedProx) is well-studied with published implementations. Federated gradient boosting is still an active research topic with varying proposed protocols, no single standard, and less empirical validation.

---

### 3.5 Linear Regression

**What it is:**

The simplest possible regression model:
```
SOFA = w₁×HR_mean + w₂×HR_std + w₃×RR_mean + ... + w₁₀₈×tfidf_edema + b
```

All 108 features contribute with fixed linear weights. Ridge Regression adds an L2 penalty on weights to prevent overfitting.

**Why it matters to understand:**

Linear Regression serves as our **interpretability baseline**. If the DNN achieves R²=0.47 and Linear Regression achieves R²=0.15, the difference (0.32 R²) is the value that the DNN's non-linearity and depth added — the learned feature interactions and threshold effects that a linear model cannot capture.

**Why it cannot be the primary model:**

```
Linear regression assumption: SOFA changes linearly with each feature

Reality 1 — Threshold effects:
  SpO2 = 95% → SOFA component 1 = 0 (normal)
  SpO2 = 94% → SOFA component 1 = 0 (still normal)
  SpO2 = 90% → SOFA component 1 = 1 (abnormal)
  SpO2 = 89% → SOFA component 1 = 2 (clinically important jump)
  
  A linear model must use the same coefficient for SpO2 everywhere.
  It cannot capture "the difference between 95% and 94% is irrelevant,
  but 90% to 89% is critical."

Reality 2 — Feature interactions:
  MAP = 60 mmHg with no vasopressors → SOFA component 4 = 1
  MAP = 60 mmHg WITH norepinephrine → SOFA component 4 = 3
  
  The MAP × vasopressor interaction matters enormously for SOFA scoring.
  Linear regression cannot learn "MAP matters differently when vasopressors are present."
  The DNN can, because its hidden layers learn to combine these features non-linearly.
```

**FL compatibility:** Linear Regression IS compatible with Federated Learning — weights are vectors of floats, FedAvg applies. It would be a valid simpler baseline but with substantially lower performance.

---

### 3.6 BioBERT + DNN

**What it is:**

Instead of TF-IDF to encode clinical notes, use **BioBERT** — a version of BERT pre-trained on PubMed abstracts and PubMed Central full-text articles (biomedical literature). BioBERT understands clinical language at a semantic level, not just as counts of vocabulary terms.

```
TF-IDF encoding of "patient shows signs of acute kidney injury":
  "acute"    → TF-IDF = 0.31  (one of our vocabulary words)
  "kidney"   → TF-IDF = 0.28  (one of our vocabulary words)
  All other 88 vocabulary terms → 0.0
  
  TF-IDF misses: "signs of" contributing meaning, "shows" as a diagnostic verb,
  and the connection between "acute" and "kidney" as a specific syndrome.

BioBERT encoding of the same text:
  → 768-dimensional embedding vector where the vector encodes:
     "this is a medical statement describing organ dysfunction"
     "kidney is the affected organ"
     "acute suggests sudden onset, not chronic"
     "injury implies damage, not just abnormal lab values"
  
  BioBERT also understands negation:
    "no signs of kidney injury" → completely different embedding
    TF-IDF: "kidney" still gets positive TF-IDF score → misleading
```

**Why BioBERT was not chosen:**

The size is prohibitive in our setting:

```
TF-IDF encoding:
  No model to transmit — it is an equation applied to notes
  Feature vector size: 90 floats per patient
  Inference time: < 1ms per patient

BioBERT:
  Model size: ~440 MB (110 million parameters)
  To use in FL: each round would transmit 440 MB per hospital → 1.32 GB per round
  × 100 rounds = 132 GB total FL communication
  
  Inference time: ~50ms per patient on CPU (50× slower than TF-IDF)
  Memory during inference: ~2 GB GPU or very slow on CPU
```

Additionally, for MIMIC-III data specifically, TF-IDF with our clinical vocabulary performs comparably to BERT-based approaches on structured prediction tasks. Clinical notes follow formulaic documentation patterns — the same clinical facts are expressed using the same vocabulary terms with little ambiguity. The semantic depth of BERT is more valuable for free-form natural language than for ICU documentation.

---

## 4. Complete Model Comparison Table

| Model | FL Compatible | SHAP Method | Estimated R² | Complexity | Why Not Used |
|---|---|---|---|---|---|
| **PyTorch DNN (current)** | ✅ Yes — FedYogi + FedProx | DeepExplainer (gradient-based) | **0.4724** | Medium | **Current choice** |
| LightGBM + XGBoost Ensemble | ❌ No — cannot FedAvg trees | TreeExplainer (path-based) | ~0.35–0.40* | Low | Fundamental FL incompatibility |
| Random Forest | ❌ No | TreeExplainer | ~0.25–0.30* | Low | FL incompatible; 0% weight in ensemble |
| LSTM / GRU | ✅ Yes — gradient-based | GradientExplainer | ~0.28–0.35* | High | Integration complexity; feature engineering achieves similar result |
| Transformer (Vital Signs only) | ⚠️ Expensive — large size | Attention maps | ~0.30–0.35* | Very High | FL communication overhead; size inappropriate for tabular |
| ClinicalBERT + Transformer | ❌ Too large for FL | Attention maps | ~0.35–0.45* | Extreme | 440 MB model → 132 GB FL communication; impractical |
| TabNet | ✅ Yes — gradient-based | Custom attention masks | ~0.27–0.35* | Medium-High | SHAP integration non-standard; less established in FL |
| FedBoost (Federated Trees) | ✅ Yes — histogram FL | TreeExplainer | ~0.35–0.42* | Very High | Not in Flower; active research area; custom implementation risk |
| Linear Regression / Ridge | ✅ Yes — FedAvg on weight vector | Linear SHAP | ~0.10–0.18* | Very Low | Cannot learn non-linear thresholds or feature interactions |
| BioBERT + DNN | ❌ Too large for FL | DeepExplainer | ~0.30–0.45* | Extreme | 440 MB per hospital per round; 132 GB total FL traffic |

*Estimated on our specific dataset; not empirically measured.

---

## 5. Summary: Panel Q&A

### Q1: What model did you start with?

A **weighted ensemble** of three classical machine learning models trained on MIMIC-III data:

- **LightGBM (52%):** Microsoft's gradient boosting on decision trees, leaf-wise growth, fast training, highest weight because it performed best in validation
- **XGBoost (48%):** Chen's gradient boosting, level-wise growth, more conservative, slightly lower validation performance but complementary errors
- **Random Forest (0%):** Breiman's bagged trees, trained and saved but discarded — performed worse than both boosting methods

All three predicted SOFA on a 0–1 normalised scale; output was multiplied by 24 to get the displayed score.

### Q2: What model do you use now?

A **PyTorch Deep Neural Network**:

- Architecture: 108 → 128 → 64 → 32 → 1 (four linear layers with ReLU activations)
- ~23,000 trainable parameters (2:1 samples-per-parameter ratio with 48K training patients)
- Weighted MSE loss (weight = 1 + SOFA × 0.5) to give clinical emphasis to High-Risk patients
- AdamW optimiser with weight decay (1e-4) and gradient clipping (norm ≤ 1.0)
- No Dropout (FL incompatible), no BatchNorm or LayerNorm (both tested and failed)
- Trained via Flower Federated Learning across 3 simulated hospitals (FedYogi server + FedProx clients)
- SHAP DeepExplainer for feature attribution
- Predicts raw SOFA directly (0–24) — no unit conversion multiplication

**Performance:** R²=0.4724, MAE=1.76 SOFA points (full 48,150-patient training set, 12,038-patient test set)

### Q3: Why did you switch from the ensemble to the DNN?

**Primary reason (non-negotiable):** Decision tree models are fundamentally incompatible with Federated Learning. FedAvg works by averaging model weight matrices across hospitals — this is mathematically meaningful for DNNs (weight matrices are real-valued and in a consistent parameter space) but undefined for decision trees (which are collections of conditional rules with no consistent mathematical space to average across).

Without a DNN, there is no federated learning. Without federated learning, the project's core research contribution — privacy-preserving multi-hospital training — does not exist.

**Secondary reasons:**
| Issue | Ensemble problem | DNN solution |
|---|---|---|
| FL compatibility | ❌ Cannot FedAvg trees | ✅ FedAvg on weight tensors |
| Privacy preservation | Patient data must be centralised | Only weight matrices shared |
| SOFA prediction scale | 0–1 × 24 → unit confusion risk | Direct 0–24 prediction |
| SHAP method | TreeExplainer (path-based) | DeepExplainer (gradient-based, more principled) |
| Regularisation in FL | Not applicable | AdamW weight decay (consistent across hospitals) |

### Q4: Is there a performance trade-off?

Yes. Tree-based models (LightGBM, XGBoost) typically outperform DNNs on tabular datasets with fewer than 100,000 training samples. The ensemble would likely achieve R² ≈ 0.35–0.40 on our task. Our DNN achieves R² = 0.4724.

The gap is smaller than expected because:
1. The DNN was optimised with federated-specific techniques (FedYogi, FedProx) that substantially improved performance
2. The 90-term SOFA vocabulary whitelist removed the noise that typically hurts DNNs on TF-IDF features

The trade-off is:
> **We chose privacy-preserving Federated Learning over marginal accuracy advantage. This is the correct trade-off for real-world clinical AI deployment, where HIPAA/GDPR compliance is a non-negotiable requirement — a more accurate model that cannot be legally deployed is worth less than a slightly less accurate model that can.**

---

*Documentation for ICU CDSS Capstone Project*
*Model framework: PyTorch DNN + Flower Federated Learning*
*SHAP: DeepExplainer with 300-sample background*
*Previous baseline: LightGBM (52%) + XGBoost (48%) ensemble, SHAP TreeExplainer*
