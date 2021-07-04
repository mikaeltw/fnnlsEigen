.PHONY: build check-pep8 clean adjust-clang-style check-clang-format check-clang-tidy test help

PYTHON = python3

build:
	git submodule update --init
	$(PYTHON) setup.py build_ext --inplace

check-pep8:
	flake8 .

adjust-clang-style:
	clang-format -i -style=file fnnlsEigen/*.hpp

check-clang-format:
	clang-format -Werror --dry-run -style=file fnnlsEigen/*.hpp

check-clang-tidy:
	clang-tidy -p . --format-style=file --extra-arg=-std=c++14 fnnlsEigen/*.hpp \
	-- -isystem thirdparty/eigen -isystem `$(PYTHON) -c "import numpy as np; print(np.get_include())"` \
	-isystem `$(PYTHON)-config --includes`

clean:
	$(PYTHON) setup.py clean
	rm -rf build
	rm -rf dist
	rm -rf .eggs
	rm -rf .cache
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf .coverage.*
	rm -rf .pytest_cache
	rm -f fnnlsEigen/eigen_fnnls.cpp
	find . -path ./env -prune -false -o -name "*.pyc" -exec rm {} \;
	find . -path ./env -prune -false -o -name "*.o" -exec rm {} \;
	find . -path ./env -prune -false -o -name "*.so" -exec rm {} \;
	find . -depth -name "__pycache__" -exec rm -rf {} \;

test:
	pytest

help:
	@echo The following targets are available:
	@echo
	@echo "  build              Build everything that needs building"
	@echo "  check-pep8         Look for errors and style violations with flake8 in the python base"
	@echo "  adjust-clang-style Autoformat the C++-header files according to the specified clang-format"
	@echo "  check-clang-format Check the C++-header files according to the specified clang-format"
	@echo "  check-clang-tidy   Check the C++-header files for possible modernisations and readability violations"
	@echo "  clean              Clean files produced during setup"
	@echo "  test               Run Python unit tests"
	@echo "  help               Show this help message"
	@echo
	@echo The default target is build.
