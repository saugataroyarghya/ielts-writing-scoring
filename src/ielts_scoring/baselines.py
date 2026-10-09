"""Quick CPU baselines, fitted on training data and reported on validation only."""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from .evaluate import regression_metrics


def surface_features(frame):
    rows = []
    for text in frame["essay"]:
        words = text.split()
        rows.append([len(words), len(text), text.count("\n"), text.count("."), text.count(","),
                     len({w.casefold() for w in words}), len(text) / max(len(words), 1)])
    return np.asarray(rows, dtype=float)


def fit_predict_targets(x_train, y_train, x_validation, alpha=10):
    predictions = []
    for i in range(y_train.shape[1]):
        mask = np.isfinite(y_train[:, i])
        regressor = Ridge(alpha=alpha, solver="lsqr")
        regressor.fit(x_train[mask], y_train[mask, i])
        predictions.append(regressor.predict(x_validation))
    return np.stack(predictions, axis=1)


def run_baselines(train, validation, targets, track, output):
    labels = train[targets].to_numpy(dtype=float)
    actual = validation[targets].to_numpy(dtype=float)
    predictions = {
        "mean": np.tile(np.nanmean(labels, axis=0), (len(validation), 1)),
        "median": np.tile(np.nanmedian(labels, axis=0), (len(validation), 1)),
    }
    scaler = StandardScaler().fit(surface_features(train))
    predictions["surface_ridge"] = fit_predict_targets(
        scaler.transform(surface_features(train)), labels, scaler.transform(surface_features(validation)),
    )
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=30000, sublinear_tf=True)),
    ])
    def texts(frame):
        return (frame["prompt"] + "\n\n" + frame["essay"]).tolist() if track == "ielts" else frame["essay"].tolist()
    predictions["tfidf_ridge"] = fit_predict_targets(
        vectorizer.fit_transform(texts(train)), labels, vectorizer.transform(texts(validation)),
    )
    metrics = {name: regression_metrics(actual, scores, targets, track == "ielts")
               for name, scores in predictions.items()}
    Path(output).write_text(json.dumps(metrics, indent=2))
    print("Validation baselines:", {name: value["macro_mae"] for name, value in metrics.items()}, flush=True)
    return metrics
