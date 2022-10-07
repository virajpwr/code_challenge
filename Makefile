# define the name of the virtual environment directory
VENV := venv

# default target, when make executed without arguments
all: venv

$(VENV)/Scripts/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/Scripts/pip install -r requirements.txt

# venv is a shortcut target
venv: $(VENV)/Scripts/activate

run: venv
	./$(VENV)/Scripts/python3 main.py

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete

.PHONY: all venv run clean