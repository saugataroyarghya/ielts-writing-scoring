"""Evaluation in original score units; no softmax or hidden score correction."""

import numpy as np
import pandas as pd


def regression_metrics(labels, predictions, targets, ielts=False):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite model predictions.")
    output = {}
    for i, name in enumerate(targets):
        mask = np.isfinite(labels[:, i])
        errors = predictions[mask, i] - labels[mask, i]
        if not len(errors):
            output[name] = {"n": 0, "mae": None, "rmse": None, "bias": None}
            continue
        row = {"n": len(errors), "mae": float(np.abs(errors).mean()),
               "rmse": float(np.sqrt((errors ** 2).mean())), "bias": float(errors.mean())}
        if ielts:
            row.update(within_half_count=int((np.abs(errors) <= 0.5).sum()),
                       within_one_count=int((np.abs(errors) <= 1).sum()))
        output[name] = row
    output["macro_mae"] = float(np.mean([r["mae"] for r in output.values() if r["mae"] is not None]))
    output["macro_rmse"] = float(np.mean([output[t]["rmse"] for t in targets if output[t]["rmse"] is not None]))
    return output


def prediction_frame(frame, labels, predictions, targets):
    output = frame[["id", "source", "group_id", "prompt_group"]].copy().reset_index(drop=True)
    output["word_count"] = frame["essay"].str.split().str.len().to_numpy()
    for i, name in enumerate(targets):
        output[f"actual_{name}"] = labels[:, i]
        output[f"predicted_{name}"] = predictions[:, i]
    return output


def error_slices(frame, labels, predictions, targets):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    errors = np.abs(predictions - labels)
    result = {}
    for key, values in {
        "prompt_group": frame["prompt_group"].astype(str).to_numpy(),
        "length": pd.cut(frame["essay"].str.split().str.len(), [0, 150, 250, 400, 700, np.inf],
                         include_lowest=True).astype(str).to_numpy(),
        "mean_score_band": np.floor(np.nanmean(labels, axis=1)).astype(int).astype(str),
    }.items():
        result[key] = {str(value): {"n": int((values == value).sum()),
                                   "mae": float(np.nanmean(errors[values == value]))}
                       for value in np.unique(values)}
    return result
