from __future__ import annotations

import math
from collections import Counter, defaultdict

import pandas as pd


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    per_class = defaultdict(lambda: {"tp": 0, "n": 0})
    for t, p in zip(y_true, y_pred):
        per_class[t]["n"] += 1
        if t == p:
            per_class[t]["tp"] += 1
    recalls = [v["tp"] / v["n"] for v in per_class.values() if v["n"] > 0]
    return sum(recalls) / len(recalls) if recalls else 0.0


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    f1s = []
    for label in labels:
        tp = sum(int(t == label and p == label) for t, p in zip(y_true, y_pred))
        fp = sum(int(t != label and p == label) for t, p in zip(y_true, y_pred))
        fn = sum(int(t == label and p != label) for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s)


def mae(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def spearmanr(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) < 2:
        return 0.0
    s1 = pd.Series(y_true).rank(method="average")
    s2 = pd.Series(y_pred).rank(method="average")
    return float(s1.corr(s2))


def quadratic_weighted_kappa(y_true: list[int], y_pred: list[int], min_rating: int = 1, max_rating: int = 5) -> float:
    if not y_true:
        return 0.0
    num_ratings = max_rating - min_rating + 1
    conf_mat = [[0 for _ in range(num_ratings)] for _ in range(num_ratings)]
    for t, p in zip(y_true, y_pred):
        conf_mat[int(t) - min_rating][int(p) - min_rating] += 1

    hist_true = Counter(y_true)
    hist_pred = Counter(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0

    expected = [[0.0 for _ in range(num_ratings)] for _ in range(num_ratings)]
    for i in range(num_ratings):
        for j in range(num_ratings):
            expected[i][j] = hist_true[min_rating + i] * hist_pred[min_rating + j] / n

    numerator = 0.0
    denominator = 0.0
    for i in range(num_ratings):
        for j in range(num_ratings):
            w = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
            numerator += w * conf_mat[i][j]
            denominator += w * expected[i][j]
    if denominator == 0:
        return 0.0
    return 1.0 - (numerator / denominator)


def classification_report(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
    }


def regression_report(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred),
    }

