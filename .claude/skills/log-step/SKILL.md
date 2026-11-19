---
name: log-step
description: Use after every checkpoint or completed step of the homework repo to append a record to docs/AI_LOG.md (what was asked, which model, what was done, files, verification, and the human-fix field as TODO(German)).
---

# log-step

Дописывай запись **в конец** `docs/AI_LOG.md` после каждого чекпойнта / завершённого шага.
Не переписывай прошлые записи.

Шаблон:

```
## YYYY-MM-DD — <шаг>
- Запрос: <что просил пользователь, одной строкой>
- Модель/агент: <Opus/Sonnet/Haiku, субагент если был>
- Что сделано: <кратко>
- Файлы: <изменённые/созданные>
- Проверка: <команды> → <результат>
- Что исправил человек: TODO(German)
```

Правила:
- Поле «Что исправил человек» всегда `TODO(German)` — заполняет только пользователь.
- В «Проверка» — реальные команды и их результат, не «должно работать».
- Без секретов, содержимого `.env` и данных из `data/`.
