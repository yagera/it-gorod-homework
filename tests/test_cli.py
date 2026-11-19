import pandas as pd
import pytest
import yaml

from itgorod.cli import main


def _t1_files(tmp_path, t1_frame, years_extra=()):
    train = t1_frame.copy()
    for y in years_extra:
        extra = t1_frame[t1_frame["year"] == 2023].copy()
        extra["year"] = y
        train = pd.concat([train, extra])
    test = t1_frame[t1_frame["year"] == 2023].drop(columns=["yearly_cv_processed"]).copy()
    test["year"] = 2024
    sample = pd.DataFrame({"department_id": test["department_id"], "yearly_cv_processed": 0})
    paths = {}
    for name, df in {"train": train, "test": test, "sample": sample}.items():
        paths[name] = str(tmp_path / f"{name}.csv")
        df.to_csv(paths[name], index=False)
    paths["output"] = str(tmp_path / "task-1.csv")
    return paths


def _t1_config(tmp_path, paths) -> str:
    cfg = {
        "seed": 42,
        "paths": paths,
        "split": {
            "folds": [
                {"train_max_year": 2021, "valid_year": 2022},
                {"train_max_year": 2022, "valid_year": 2023},
            ]
        },
        "target_year": 2024,
        "placeholder": {"fallback": "tech_stack_median"},
    }
    path = tmp_path / "t1.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_validate_ok_returns_0(tmp_path, t1_frame, capsys):
    paths = _t1_files(tmp_path, t1_frame)
    config = _t1_config(tmp_path, paths)
    assert main(["t1", "predict", "--config", config]) == 0
    assert main(["validate", "t1", paths["output"], "--config", config]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_bad_returns_1(tmp_path, t1_frame, capsys):
    paths = _t1_files(tmp_path, t1_frame)
    config = _t1_config(tmp_path, paths)
    bad = pd.read_csv(paths["sample"])  # все значения 0 → нарушение
    bad.to_csv(paths["output"], index=False)
    assert main(["validate", "t1", paths["output"], "--config", config]) == 1
    assert "yearly_cv_processed" in capsys.readouterr().out


def test_t2_train_predict_without_smoke_returns_2(tmp_path, capsys):
    assert main(["t2", "train-predict", "--config", "configs/t2.yaml"]) == 2
    assert "этапе T2" in capsys.readouterr().out


def test_check_time_split_ok(tmp_path, t1_frame, capsys):
    config = _t1_config(tmp_path, _t1_files(tmp_path, t1_frame))
    assert main(["check-time-split", "--config", config]) == 0
    assert capsys.readouterr().out.count("OK") >= 2  # по строке на фолд


def test_check_time_split_future_year_returns_1(tmp_path, t1_frame):
    config = _t1_config(tmp_path, _t1_files(tmp_path, t1_frame, years_extra=(2025,)))
    assert main(["check-time-split", "--config", config]) == 1


def test_no_command_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
