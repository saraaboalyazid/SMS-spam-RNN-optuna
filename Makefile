# -----------------------------
# Makefile for RNN Project
# -----------------------------

# Variables
PYTHON = ../venv312/Scripts/python.exe
SRC_DIR = .
DATA_DIR = $(SRC_DIR)/data
EXPERIMENTS_DIR = $(SRC_DIR)/experiments
MODEL_DIR = $(SRC_DIR)/models

# -----------------------------
# Commands
# -----------------------------

setup:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt

tune:
	@echo "Running hyperparameter tuning with Optuna..."
	$(PYTHON) $(EXPERIMENTS_DIR)/train_with_optuna.py

train_final:
	@echo "Training final model with best Optuna hyperparameters..."
	$(PYTHON) $(EXPERIMENTS_DIR)/train_final_model.py

clean:
	@echo "Cleaning temporary files..."
	-del /Q /F *.pyc 2>nul
	-del /Q /F *.pyo 2>nul
	-del /Q /F *.log 2>nul
	-del /Q /F *.pth 2>nul
	-del /Q /F *.json 2>nul
	-rmdir /S /Q __pycache__ 2>nul

help:
	@echo ""
	@echo "Available commands:"
	@echo "  make setup        - Install dependencies"
	@echo "  make tune         - Run Optuna hyperparameter tuning"
	@echo "  make train_final  - Train final model with best params"
	@echo "  make clean        - Clean up temp files"
	@echo ""
