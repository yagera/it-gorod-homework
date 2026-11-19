.PHONY: install lint format test t1 t2-prepare-smoke t2-smoke validate check-time-split ci

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

t1:
	uv run itgorod t1 predict --config configs/t1.yaml

t2-prepare-smoke:
	uv run itgorod t2 prepare --config configs/t2.yaml --smoke

t2-smoke:
	uv run itgorod t2 train-predict --config configs/t2.yaml --smoke

validate:
	uv run itgorod validate t1 outputs/task-1.csv
	uv run itgorod validate t2 outputs/task-2.smoke.csv

check-time-split:
	uv run itgorod check-time-split --config configs/t1.yaml

ci: lint test
