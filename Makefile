# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

NAME := adrl
PACKAGE_NAME := adrl

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
.PHONY: help install-dev check format pre-commit clean help
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	PYTHON ?= python
PIP ?= python -m pip
MAKE ?= make
BLACK ?= black
PRECOMMIT ?= pre-commit
FLAKE8 ?= flake8

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

check-black:
	$(BLACK) ${SOURCE_DIR} --check || :
	check-flake8:
	$(FLAKE8) ${SOURCE_DIR} || :
	$(FLAKE8) ${TESTS_DIR} || :

check: check-black check-flake8

pre-commit:
	$(PRECOMMIT) run --all-files

format-black:
	$(BLACK) ${SOURCE_DIR}
	format: format-black

# Clean up any builds in ./dist as well as doc, if present
clean: 