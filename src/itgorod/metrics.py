"""Метрики заданий: единое место для MAPE (T1) и Macro-F1 (T2)."""

from sklearn.metrics import f1_score, mean_absolute_percentage_error


def mape(y_true, y_pred) -> float:
    """MAPE как доля (0.1 = 10%)."""
    return float(mean_absolute_percentage_error(y_true, y_pred))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
