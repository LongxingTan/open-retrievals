.PHONY: style test docs pre-release

check_dirs := src examples tests


style: ## run checks on all files and potentially modifies some of them
	black $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)
	pre-commit run --all-files


test: ## run tests for the library
	python -m unittest


docs:  ## run tests for the docs
	make -C docs clean M=$(shell pwd)
	make -C docs html M=$(shell pwd)


help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[33m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z\/_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
