# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

NAME := adrl
PACKAGE_NAME := adrl

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
.PHONY: help install-dev install check format pre-commit
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	PYTHON ?= python
PIP ?= python -m pip
MAKE ?= make
RUFF ?= python -m ruff

install-dev:
	$(PIP) install -e ".[dev]"
	$(PIP) install minari
	$(PIP) install dacbench
	$(PIP) install gymnasium==0.29.1
	$(PIP) install gymnasium-robotics>=1.2.3

install:
	$(PIP) install -e .
	$(PIP) install minari
	$(PIP) install dacbench
	$(PIP) install gymnasium==0.29.1
	$(PIP) install gymnasium-robotics>=1.2.3

check: 
	$(RUFF) check --exit-zero $(SOURCE_DIR)

format:
	$(RUFF) check --silent --exit-zero --no-cache --fix $(SOURCE_DIR)
	$(RUFF) format $(SOURCE_DIR)