.DEFAULT_GOAL := help

RUN := poetry run
PYTHON := $(RUN) python

.PHONY: install
install: ## Install dependencies
	poetry install

.PHONY: preprocess
preprocess: ## Preprocess data
	$(PYTHON) scripts/preprocess.py $(ARGS)

.PHONY: train
train: ## Run centralized/local training
	$(PYTHON) scripts/train.py $(ARGS)

.PHONY: federated-train
federated-train: ## Run Flower federated simulation
	$(PYTHON) scripts/federated_train.py $(ARGS)

.PHONY: federated-server
federated-server: ## Start Flower server
	$(PYTHON) scripts/federated_server.py $(ARGS)

.PHONY: federated-client
federated-client: ## Start Flower client
	$(PYTHON) scripts/federated_client.py $(ARGS)

.PHONY: evaluate
evaluate: ## Run closed/open-set evaluation
	$(PYTHON) scripts/evaluate.py $(ARGS)

.PHONY: plot
plot: ## Regenerate plots from a run directory
	$(PYTHON) scripts/plot.py $(ARGS)

.PHONY: compare-runs
compare-runs: ## Compare run metrics
	$(PYTHON) scripts/compare_runs.py $(ARGS)

.PHONY: smoke
smoke: ## Run synthetic smoke test
	$(PYTHON) scripts/smoke_test.py experiment=smoke $(ARGS)

.PHONY: test
test: ## Run tests
	$(PYTHON) -m pytest -q

.PHONY: help
help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
