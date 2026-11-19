"""Шпионы для правила A: фиксируют КАЖДЫЙ fit векторайзера и модели (fit, fit_transform,
partial_fit) — что именно им передали. Чужие данные в любом fit → тест красный."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


class Calls:
    def __init__(self):
        self.vec_fits: list[list[str]] = []  # документы каждого fit/fit_transform векторайзера
        self.model_fits: list[tuple[int, list]] = []  # (число строк X, метки y) каждого fit модели


def make_spies(calls: Calls):
    class SpyVec(TfidfVectorizer):
        def fit(self, raw_documents, y=None):
            raw_documents = list(raw_documents)
            calls.vec_fits.append(raw_documents)
            return super().fit(raw_documents, y)

        def fit_transform(self, raw_documents, y=None):
            raw_documents = list(raw_documents)
            calls.vec_fits.append(raw_documents)
            return super().fit_transform(raw_documents, y)

    class SpyModel(LinearSVC):
        def fit(self, X, y, **kw):
            calls.model_fits.append((X.shape[0], list(y)))
            return super().fit(X, y, **kw)

        def partial_fit(self, X, y, *a, **kw):  # у LinearSVC нет: любой вызов — нарушение
            calls.model_fits.append((X.shape[0], list(y)))
            raise AssertionError("partial_fit не ожидается")

    return SpyVec, SpyModel
