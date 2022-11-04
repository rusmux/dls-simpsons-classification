.PHONY: clean create_environment format get_data install install_dev\
lint make_requirements make_requirements_dev test_environment

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = dls-simpsons-classification

RAW_DATA_DIR = data/raw

PYTHON_VERSION = 3.8

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

ifeq (,$(shell which mamba))
HAS_MAMBA=False
else
HAS_MAMBA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
install: test_environment
	pip install .
	jupyter contrib nbextension install --user

## Install Python development dependencies
install_dev: make_requirements_dev
	pip install -r requirements.dev.txt
	rm -rf requirements.dev.txt

## Make requirements.txt from pyproject.toml
make_requirements: test_environment
	poetry export -f requirements.txt -o requirements.txt --without-hashes

## Make development requirements.txt from pyproject.toml
make_requirements_dev: test_environment
	poetry export -f requirements.txt -o requirements.dev.txt --only=dev --without-hashes

## Get data from Kaggle
get_data:
	mkdir -p $(RAW_DATA_DIR)
	mkdir -p $(RAW_DATA_DIR)/test/test

	kaggle competitions download -p $(RAW_DATA_DIR) -c journey-springfield
	unzip -n -q $(RAW_DATA_DIR)/journey-springfield.zip -d $(RAW_DATA_DIR)

	rm -rf $(RAW_DATA_DIR)/journey-springfield.zip
	rm -rf $(RAW_DATA_DIR)/characters_illustration.png $(RAW_DATA_DIR)/sample_submission.csv

	mv $(RAW_DATA_DIR)/train/simpsons_dataset/* $(RAW_DATA_DIR)/train
	mv $(RAW_DATA_DIR)/testset/testset/* $(RAW_DATA_DIR)/test/test

	rm -rf $(RAW_DATA_DIR)/train/simpsons_dataset $(RAW_DATA_DIR)/testset

## Delete all unwanted files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete


## Format
format:
	black .
	isort .
	nbqa black .
	nbqa isort .

## Lint
lint:
	pylint --exit-zero src
	flake8 --exit-zero .
	mypy .

## Set up Python interpreter environment
create_environment:
ifeq (True,$(HAS_MAMBA))
		@echo "Detected mamba, creating mamba environment"
	mamba create --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
else
	ifeq (True,$(HAS_CONDA))
			@echo "Detected conda, creating conda environment"
		conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	else
		pip install -q virtualenv virtualenvwrapper
		@echo "Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
		export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
		@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=python"
		@echo "New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	endif
endif

## Test Python environment is setup correctly
test_environment:
	python test_environment.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
