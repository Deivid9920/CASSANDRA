"""Probe training contract: for every (model, layer, mode, type)
combination a StandardScaler + probe (logistic over C_grid, or
MLP with mlp_hidden units and early stopping) is trained on 4
folds and evaluated on the held-out fold; the scaler is fitted
inside each training fold, never globally. All out-of-fold
predictions are persisted to logs/oof_predictions.jsonl."""
