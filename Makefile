.PHONY: install test lint format typecheck run all

install:
	pip install -e ".[dev]"

test:
	pytest --cov=codesentinel tests/

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy codesentinel/ app.py

run:
	streamlit run app.py

all: format lint typecheck test
