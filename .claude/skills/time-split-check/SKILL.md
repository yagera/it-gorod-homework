---
name: time-split-check
description: Use whenever T1 features, targets, validation folds or the training window change, to check that no future-year information leaks into a forecast (rule A).
---

# time-split-check

## Правило A
Для прогноза года Y можно использовать все признаки года Y и любые данные лет < Y (включая лаги таргета). **Нельзя:** таргет года Y и любые данные лет > Y. Валидация T1 — только по времени, фолды в `configs/t1.yaml` (`split.folds`, expanding window: ≤2021→2022 и ≤2022→2023).

## Автоматические проверки
```
make check-time-split
uv run pytest tests/test_time_split.py -v
```
Первая команда проверяет оба фолда и признаки 2024 на реальных данных (`LEAKAGE:` + список = нарушение). Тест параметризован по фолдам из конфига.

## Ручной чеклист (то, что автоматика не видит)
- Лаги и скользящие признаки считаются `groupby(department_id).shift(...)`, а не по всей таблице.
- Агрегаты и таргет-энкодинги по отделу/`tech_stack` считаются только по train-годам фолда.
- Нет `fit` (скейлер, импьютер, модель) на valid или test; импутация пропусков — по train-годам.
- Нет случайного K-fold и перемешивания.
- Новый признак не использует таргет целевого года; все признаки нового кода проходят через `assert_features_for_year`.
- Подбор гиперпараметров и early stopping не смотрят на valid-год, которым потом меряют качество (иначе оценка завышена).
