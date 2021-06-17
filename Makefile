.PHONY: build check clean test help

PYTHON = python3

build:
	git submodule update --init
	$(PYTHON) setup.py build_ext --inplace

check:
	flake8 .

clean:
	$(PYTHON) setup.py clean
	rm -rf build
	rm -rf dist
	rm -rf .eggs
	rm -rf .cache
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf .coverage.*
	rm -f src/eigen_fnnls.cpp
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.o" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find . -depth -name "__pycache__" -exec rm -rf {} \;

test:
	pytest

help:
	@echo The following targets are available:
	@echo
	@echo "  build              Build everything that needs building"
	@echo "  test               Run Python unit tests"
	@echo "  check              Look for errors and style violations with flake8"
	@echo "  clean              Clean files produced during setup"
	@echo "  help               Show this help message"
	@echo
	@echo The default target is build.
