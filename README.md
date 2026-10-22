# House Price Prediction

Predicting residential sale prices with five tuned regression models — Random Forest, Gradient Boosting, XGBoost, LightGBM and a neural network — and comparing them head to head.

## Overview

The script walks through a full modelling workflow on a King County–style housing dataset:

1. **Explore** — missing-value check, correlation heatmap, price distribution.
2. **Engineer features** — sale year and month, house age, years since renovation.
3. **Prepare** — 70/30 train–test split, standard scaling fitted on the training set only.
4. **Tune and train** — 5-fold `GridSearchCV` for each model.
5. **Evaluate** — MSE, MAE and R² on the held-out test set, plus Random Forest feature importances and residuals.

## Models

| Model | Tuned parameters |
| --- | --- |
| Random Forest | `n_estimators`, `max_depth` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth` |
| LightGBM | `n_estimators`, `learning_rate`, `num_leaves` |
| MLP Regressor | `hidden_layer_sizes`, `activation`, `learning_rate` |

## Design note: avoiding target leakage

An earlier version used `price_per_sqft` (price ÷ living area) as a feature. Because it is computed from the target, it lets the model see the answer and makes the scores look far better than they really are. It has been removed, so the reported metrics reflect genuine predictive performance.

## Getting started

```bash
git clone https://github.com/kr-xna/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

Place the dataset in the project folder as `Part1_house_price.csv`, then run:

```bash
python house_price_prediction.py
```

The dataset is not included in this repository. It expects the standard King County columns (`id`, `date`, `price`, `sqft_living`, `yr_built`, `yr_renovated`, `lat`, `long`, …).

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Matplotlib · seaborn
