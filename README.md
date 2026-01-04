# LA Crime Classification — From Perceptron to Neural Network (MLP)

> **Task:** Binary classification of LAPD crimes into **Part 1 vs Part 2** using LA Crime Data (2020–Present).  
> **Project arc:** Start with a **linear Perceptron baseline**, then extend to a **non-linear Neural Network (MLP)** while keeping the same preprocessing pipeline for a fair comparison.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Definition](#problem-definition)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models](#models)
  - [Perceptron Baseline](#perceptron-baseline)
  - [Neural Network (MLP)](#neural-network-mlp)
- [Results](#results)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Visual Highlights](#visual-highlights)
- [Reproducibility](#reproducibility)
- [Recommended Improvements](#recommended-improvements)
- [Repository Structure](#repository-structure)
- [Credits](#credits)

---

## Project Overview

This repository is an applied machine learning project built around a real-world, high-volume tabular dataset: LAPD crime incidents from 2020 to present.  
The project is designed to be both:
- **educational** (how a Perceptron works, and where it fails), and
- **practical** (a full preprocessing + model + evaluation workflow on messy real data).

The key design choice: **keep preprocessing fixed**, and swap models.  
That lets us make a controlled claim:

> If performance improves, it’s because the model learned better structure — not because we changed the data pipeline.

---

## Problem Definition

We model crime severity as a **binary classification** task:

- **Class “1”**: Part 1 crimes (more severe)
- **Class “2”**: Part 2 crimes (less severe)

The goal is to predict the class using temporal, spatial, demographic, and contextual features.

Primary evaluation metric: **ROC-AUC**  
(ROC-AUC is threshold-independent and more reliable than accuracy when classes are imbalanced.)

---

## Dataset

- **Source**: LAPD Crime Data (2020–Present) CSV (`Crime_Data_from_2020_to_Present.csv`)
- **Scale**: ~1M incidents (large, noisy, missing values, mixed types)
- **Coverage**: Los Angeles area

> **Note:** The raw CSV is not committed to this repo due to size.

---

## Feature Engineering

This project constructs model-ready features from raw columns:

### Temporal
- Year, Month, Weekday
- Hour, Minute
- IsWeekend

### Spatial
- Latitude/Longitude binning (to reduce noise + create locality features)
- Area Name

### Demographic
- Victim age
- Victim sex

### Contextual
- Premise description
- Weapon description

These feature groups are intentionally heterogeneous — which is why the preprocessing is built as separate numeric vs categorical pipelines.

---

## Preprocessing Pipeline

A single sklearn **ColumnTransformer** is used for both models to ensure fairness and prevent leakage.

### Numeric pipeline
- Missing values → median imputation
- Scaling → StandardScaler

### Categorical pipeline
- Missing values → most frequent imputation
- One-hot encoding (`handle_unknown="ignore"`)
- Optional: minimum-frequency grouping to reduce sparse explosion

---

## Models

### Perceptron Baseline
A Perceptron is a linear classifier. It learns a single decision boundary:

\[
\hat{y} = \text{sign}(w^T x + b)
\]

Pros:
- Fast, simple baseline
- Good for understanding linear separability

Limitations:
- Cannot learn non-linear interactions (e.g., hour × area × premise)
- Often underfits complex tabular patterns

The repo originally implemented:
- hyperparameter optimization
- probability calibration
- standard evaluation + diagnostics

---

### Neural Network (MLP)

We extend the Perceptron into a **Multi-Layer Perceptron (MLP)** using TensorFlow/Keras, integrated into the sklearn pipeline using **SciKeras**.

#### Why this is a “natural” extension
A neural network is essentially stacked perceptrons with non-linear activations:

- Hidden layers (Dense + ReLU) learn feature interactions
- Dropout reduces overfitting
- Sigmoid output gives probability for binary classification

#### Integration design (important)
We keep the same pipeline structure:

**Preprocess (ColumnTransformer) → Model**

This ensures the comparison is about **model capacity**, not data differences.

---

## Results

### Perceptron (baseline)
- **Accuracy:** 0.7532
- **ROC-AUC:** 0.8349
- **Macro F1:** 0.7318

### Neural Network (MLP)
- **Accuracy:** ~0.81
- **ROC-AUC:** ~0.89
- Class-wise performance improves, especially in overall separability (ROC-AUC)

### Comparison Summary

| Model | ROC-AUC | Accuracy | Macro F1 |
|------|--------:|---------:|---------:|
| Perceptron | 0.835 | 0.753 | 0.732 |
| Neural Network (MLP) | **0.89** | **0.81** | **~0.80** |

**Interpretation:**  
The MLP improves ROC-AUC substantially, indicating better separation between Part 1 vs Part 2 crimes across thresholds. This supports the conclusion that the task is **not linearly separable**, and benefits from non-linear modeling.

---

## Notebook Walkthrough

The primary workflow is contained in the notebook:

- **EDA & cleaning**: missing values, validity checks, distributions
- **Feature engineering**: time parsing, bins, derived columns
- **Preprocessing pipeline**: ColumnTransformer for numeric + categorical
- **Model 1**: Perceptron baseline (+ tuning/calibration)
- **Model 2**: Neural Network MLP (SciKeras inside sklearn Pipeline)
- **Evaluation**: ROC-AUC, classification report, confusion matrix, ROC curve
- **Diagnostics**: permutation importance (optional), error patterns

---

## Visual Highlights

Recommended visuals to keep the notebook and README “report-like”:

- Crimes by hour and weekday (temporal rhythms)
- Top crime categories; area distribution
- Confusion matrix and ROC curve (evaluation)
- Permutation importance bar chart (top features)
- (Optional) Map of LA crimes (geopandas + contextily)

> Tip: Save figures to `reports/figures/` and embed them in the README.

---

## Reproducibility

### 1) Create environment
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
