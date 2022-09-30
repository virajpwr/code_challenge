.PHONY: help lint run

# Makefile variables
VENV_NAME:=venv
PYTHON=${VENV_NAME}/bin/python3

# Include your variables here
RANDOM_SEED:=42
NUM_EPOCHS:=15
INPUT_DIM:=784
HIDDEN_DIM:=128
OUTPUT_DIM:=10

.DEFAULT: help
help:
	@echo "make venv"
	@echo "       prepare development environment, use only once"
	@echo "make lint"
	@echo "       run pylint"
	@echo "make run"
	@echo "       run project"

# Comment this when using Linux
venv: $(VENV_NAME)/Scripts/activate
$(VENV_NAME)/Scripts/activate: setup.py
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -e .
	rm -rf ./*.egg-info
	touch $(VENV_NAME)/Scripts/activate

# Comment this when using windows
# venv: $(VENV_NAME)/bin/activate
# $(VENV_NAME)/bin/activate: setup.py
# 	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
# 	${PYTHON} -m pip install -U pip
# 	${PYTHON} -m pip install -e .
# 	rm -rf ./*.egg-info
# 	touch $(VENV_NAME)/bin/activate

lint: venv
	${PYTHON} -m pylint main.py

run: venv
	${PYTHON} main.py