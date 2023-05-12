SHELL = /bin/bash

Base_PATH := $(CURDIR)

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv			: creates a virtual environment."
	@echo "style			: executes style formatting."
	@echo "clean			: cleans all unnecessary files."
	@echo "test			: execute tests on code, data and models."
	@echo "dvc_data		: only push data to dvc s3 storage."
	@echo "dvc_mlflow		: only push latest mlflow logs."
	@echo "dvc_find_push_model	: finds and pushes artifacts of new model created within 2 days."
	@echo "print_files_for_push	: prints all file paths the that needs to be pushed remote DVC storage."
	@echo "dvc_push		: pushes all file paths the that needs to be pushed remote DVC storage."

# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .

# Environment
.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install --upgrade pip setuptools wheel && \
	python3 -m pip install -e ".[dev]" && \
	pre-commit install && \
	pre-commit autoupdate

# Cleaning
.PHONY: clean
clean: clean
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

# Test
.PHONY: test
test:
	pytest -m "not training"
	cd tests && great_expectations checkpoint run labels

# only data push
.PHONY: dvc_data
dvc_data:
	dvc add data/labels.csv && dvc add data/images && dvc push -r s3_data

# push all mlflow logs
.SILENT: dvc_mlflow
.PHONY: dvc_mlflow
dvc_mlflow:
	LATEST_EXP_DIR=$$(ls -td outputs/mlflow/Experiments/*/ | head -n 1); \
	printf "%s\n" "$$LATEST_EXP_DIR"; \
	dvc add -R $$LATEST_EXP_DIR && dvc push -r mlflow_logs; \
	LATEST_OPT_DIR=$$(ls -td outputs/mlflow/Optimization_Tasks/*/ | head -n 1); \
	printf "%s\n" "$$LATEST_OPT_DIR"; \
	dvc add -R $$LATEST_OPT_DIR && dvc push -r mlflow_logs


# push new model only which changed within 2 days
.SILENT: dvc_find_push_model
.PHONY: dvc_find_push_model
dvc_find_push_model:
	for run_dir in $$(find outputs/model/Experiments -mindepth 2 -maxdepth 2 -type d -mtime -2); do \
		printf "%s\n" "$$run_dir"; \
		dvc add $$run_dir && dvc push -r s3_model; \
	done
	for run_dir in $$(find outputs/model/Optimization_Tasks -mindepth 1 -maxdepth 1 -type d -mtime -2); do \
		printf "%s\n" "$$run_dir"; \
		dvc add $$run_dir && dvc push -r s3_model; \
	done


.SILENT: print_files_for_push
.PHONY: print_files_for_push
print_files_for_push:
	for run_dir in $$(find outputs/model/Experiments -mindepth 2 -maxdepth 2 -type d -mtime -2); do \
		printf "%s\n" "$$run_dir"; \
	done
	for run_dir in $$(find outputs/model/Optimization_Tasks -mindepth 1 -maxdepth 1 -type d -mtime -2); do \
		printf "%s\n" "$$run_dir"; \
	done
	printf "%s\n" $$(ls -td outputs/mlflow/Experiments/*/ | head -n 1)
	printf "%s\n" $$(ls -td outputs/mlflow/Optimization_Tasks/*/ | head -n 1)



# push all of them to the dvc
.SILENT: dvc_push
.PHONY: dvc_push
dvc_push: dvc_find_push_model dvc_data dvc_mlflow
