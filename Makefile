.PHONY: install lint format test t1 t1-experiment t2-prepare-smoke t2-smoke t2-experiment t2-tune t2-tune2 t2-stability t2-swapdate t2-embed t2-embed-eval t2 validate check-time-split ci submit validate-submit

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

t1-experiment:
	uv run itgorod t1 experiment --config configs/t1.yaml

t2-prepare-smoke:
	uv run itgorod t2 prepare --config configs/t2.yaml --smoke

t2-smoke:
	uv run itgorod t2 train-predict --config configs/t2.yaml --smoke

t2:
	uv run itgorod t2 train-predict --config configs/t2.yaml

t2-tune:
	uv run itgorod t2 tune --config configs/t2.yaml

t2-tune2:
	uv run itgorod t2 tune2 --config configs/t2.yaml

t2-stability:
	uv run itgorod t2 stability --config configs/t2.yaml

t2-swapdate:
	uv run itgorod t2 swapdate --config configs/t2.yaml

t2-embed:
	uv run itgorod t2 embed --config configs/t2.yaml

t2-embed-eval:
	uv run itgorod t2 embed-eval --config configs/t2.yaml

t2-experiment:
	uv run itgorod t2 experiment --config configs/t2.yaml

validate:
	uv run itgorod validate t1 outputs/task-1.csv
	uv run itgorod validate t2 outputs/task-2.smoke.csv
	uv run itgorod validate t2 outputs/task-2.csv

check-time-split:
	uv run itgorod check-time-split --config configs/t1.yaml

ci: lint test

validate-submit:
	uv run itgorod validate t1 outputs/task-1.csv
	uv run itgorod validate t2 outputs/task-2.csv

submit: t1 t2
	$(MAKE) validate-submit
