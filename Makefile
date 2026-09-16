# NOTICE: This file is new in this fork and does not exist in the original
# google/meridian source.
#
# One-click entry points around the existing scripts/ tooling. This file does
# not reimplement scripts/setup.sh, scripts/verify_environment.py or
# scripts/compile_report_css.py -- it just wraps them so a newcomer can run
# `make` instead of reading the README top to bottom.
#
# Works on macOS (Apple Silicon) and Linux. Assumes only `bash`, `make`,
# `python3` and standard POSIX tools -- no GNU-only flags.

SHELL := /bin/bash

# Override with `make VENV=~/.venvs/mmm setup` etc.
VENV ?= .venv
# Make does not perform shell tilde expansion after a variable is substituted
# into a quoted recipe argument, so expand it while evaluating the Makefile.
override VENV := $(subst ~,$(HOME),$(VENV))
VENV_PY := $(VENV)/bin/python
TEST_WORKERS ?= 2

.DEFAULT_GOAL := help

.PHONY: help setup verify test test-tf test-e2e demo css build clean distclean all quickstart check-venv

## help: Show this list of targets.
help:
	@echo "Meridian -- available targets:"
	@grep -E '^## [a-zA-Z0-9_-]+:' $(MAKEFILE_LIST) | sed -E 's/^## /  /' | \
		awk -F':' '{printf "  %-12s%s\n", $$1, $$2}'

# Internal guard: most targets need a working venv. Fails fast with a clear
# message instead of a confusing "no such file or directory" from python3.
check-venv:
	@if [ ! -x "$(VENV_PY)" ]; then \
		echo "No virtual environment at '$(VENV)' yet. Run 'make setup' first." >&2; \
		exit 1; \
	fi

## setup: Build the venv, install supported CPU extras, compile CSS, verify (wraps scripts/setup.sh).
setup:
	./scripts/setup.sh "$(VENV)"

## verify: Check that the environment is actually usable (scripts/verify_environment.py).
verify: check-venv
	"$(VENV_PY)" scripts/verify_environment.py

## test: Run the full suite in fresh processes with bounded worker memory.
test: check-venv
	"$(VENV_PY)" scripts/run_tests.py --workers "$(TEST_WORKERS)"

## test-tf: Run the full test suite with the TensorFlow backend forced.
test-tf: check-venv
	MERIDIAN_BACKEND=tensorflow "$(VENV_PY)" scripts/run_tests.py --workers "$(TEST_WORKERS)"

## test-e2e: Run the real fit → optimize → serialize → report smoke test.
test-e2e: check-venv
	"$(VENV_PY)" scripts/compile_report_css.py
	"$(VENV_PY)" scripts/test_end_to_end.py

## demo: Run the quickstart end-to-end example (examples/quickstart.py).
demo: check-venv
	"$(VENV_PY)" examples/quickstart.py

## css: Compile the report stylesheet (scripts/compile_report_css.py).
css: check-venv
	"$(VENV_PY)" scripts/compile_report_css.py

## build: Build the wheel and sdist into dist/.
build: check-venv
	"$(VENV_PY)" -m pip install -q -U build
	"$(VENV_PY)" -m build

## clean: Remove build artifacts and generated output. Never touches the venv.
clean:
	rm -rf build dist *.egg-info google_meridian.egg-info proto/dist proto/*.egg-info
	rm -rf quickstart_output
	rm -f meridian/templates/style.css
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -not -path './.venv/*' -exec rm -rf {} +
	@echo "Cleaned build artifacts and generated output. '$(VENV)' was left untouched."

## distclean: clean, then also delete the venv. Asks for confirmation first.
distclean: clean
	@echo "This will permanently delete '$(VENV)' (the entire virtual environment)."
	@read -r -p "Continue? [y/N] " reply; \
		case "$$reply" in \
			[yY]|[yY][eE][sS]) rm -rf "$(VENV)"; echo "Removed '$(VENV)'." ;; \
			*) echo "Aborted. '$(VENV)' was left untouched." ;; \
		esac

## quickstart: The one-click path -- setup, then run the demo.
quickstart: setup
	$(MAKE) VENV="$(VENV)" demo

## all: Alias for quickstart.
all: quickstart
