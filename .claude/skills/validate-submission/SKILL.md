---
name: validate-submission
description: Use after creating or changing any submission CSV in outputs/ (task-1.csv, task-2.csv, task-2.smoke.csv) to check its format against sample_submission before reporting it as done.
---

# validate-submission

Запусти:

```
uv run itgorod validate t1 outputs/task-1.csv
uv run itgorod validate t2 outputs/task-2.csv        # или outputs/task-2.smoke.csv
make validate                                        # task-1.csv и task-2.smoke.csv
```

Код возврата: 0 = `OK`, 1 = список нарушений формата, 2 = проверка не выполнена (нет файла сабмишна, sample или train; в выводе `ERROR: file not found: <путь>`). Реализация — `itgorod/validation.py` (те же функции в pytest).

## Правила
**T1** (`department_id,yearly_cv_processed`): ровно эти колонки; число строк равно числу строк в `sample_submission.csv` из конфига (для выданного sample — 505); порядок `department_id` как в `sample_submission.csv`, без дублей; значения конечные и > 0.

**T2** (`row_id,code`): ровно эти колонки; число строк равно числу строк в sample (для выданного sample — 10001); `row_id` как в sample (по порядку); `code` — строка вида `NNN.NNN.NNN` (только ASCII-цифры) и только из кодов train; пропусков нет.

## Типовые ошибки
- Лишняя колонка `Unnamed: 0` → писать `to_csv(index=False)`.
- `code` без ведущих нулей / не строка → читать и писать только как `str` (`dtype=str`).
- Код не из train → предсказание вне 58 классов; ограничить модель кодами train.
- Нули или NaN в T1 → у отдела нет ни одного значения; проверить фолбэки.
- Порядок строк → переиндексировать по sample (`merge` на id слева от sample).

Валидатор сверяет сабмишн с sample из конфига: убедись, что `paths.sample` указывает на выданный `sample_submission.csv`, иначе проверка числа строк бессмысленна.

Не называй сабмишн готовым к сдаче, пока валидатор не вернул `OK`. `task-2.smoke.csv` — только проверка формата, не результат.
