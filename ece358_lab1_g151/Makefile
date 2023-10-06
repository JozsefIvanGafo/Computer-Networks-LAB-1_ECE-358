# Makefile for building and running main.py

# Define the Python interpreter (you can change this if needed)
PYTHON = python

# Define the name of your Python script
SCRIPT = main.py

# Define the requirements.txt
REQUIREMENTS_FILE = requirements.txt

# Default target: build and run the script
all: build run

# Target for installing the requirements
install:
	@echo "Install $(REQUIREMENTS_FILE)"
	pip install -r $(REQUIREMENTS_FILE)

# Target for building the script (if needed)
build:
	@echo "Building $(SCRIPT)"

# Target for running the script
run:
	@echo "Running $(SCRIPT)"
	$(PYTHON) $(SCRIPT)

# Target for cleaning up generated files (if needed)
clean:
	@echo "Cleaning up"

.PHONY: all build run clean
