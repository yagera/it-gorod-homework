import pytest

from itgorod.metrics import macro_f1, mape


def test_mape_is_a_fraction():
    assert mape([100, 200], [110, 180]) == pytest.approx(0.1)


def test_macro_f1():
    assert macro_f1(["a", "b"], ["a", "a"]) == pytest.approx(1 / 3)
