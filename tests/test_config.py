import numpy as np
import pytest

from itgorod.config import Fold, load_t1_config, load_t2_config
from itgorod.seed import set_seed


def test_t1_config_loads():
    cfg = load_t1_config("configs/t1.yaml")
    assert cfg.seed == 42
    assert cfg.folds == [Fold(2021, 2022), Fold(2022, 2023)]
    assert cfg.target_year == 2024
    assert cfg.paths.train.is_absolute()


def test_t2_config_loads():
    cfg = load_t2_config("configs/t2.yaml")
    assert cfg.max_chars == 5000
    assert cfg.smoke_n_train_rows == 5000
    assert cfg.paths.smoke_output.name == "task-2.smoke.csv"


def test_unknown_key_rejected(tmp_path):
    src = open("configs/t1.yaml", encoding="utf-8").read()
    bad = tmp_path / "t1.yaml"
    bad.write_text(src + "foo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="foo"):
        load_t1_config(bad)


def test_set_seed_deterministic():
    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    assert (a == np.random.rand(3)).all()


def test_unknown_key_in_fold_rejected(tmp_path):
    src = open("configs/t1.yaml", encoding="utf-8").read()
    bad = tmp_path / "t1.yaml"
    bad.write_text(
        src.replace(
            "{train_max_year: 2022, valid_year: 2023}",
            "{train_max_year: 2022, valid_year: 2023, bar: 1}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bar.*split\.folds\[1\]"):
        load_t1_config(bad)
