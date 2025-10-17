# -----------------------------
# Makefile for RNN Project
# -----------------------------

# Variables
PYTHON = ../venv312/Scripts/python.exe
SRC_DIR = .
DATA_DIR = $(SRC_DIR)/data
EXPERIMENTS_DIR = $(SRC_DIR)/experiments
MODEL_DIR = $(SRC_DIR)/models
IMAGE_NAME = sms-spam-rnn
ECR_REGISTRY = public.ecr.aws/k5v8x8j7
TAG = latest
EXCLUDE_DIRS=venv,venv312,__pycache__,.git,.github

PYTHON=python

# -----------------------------
# Commands
# -----------------------------

setup:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r ../requirements.txt
lint:
	@echo "Running linter..."
	flake8 . --exclude=venv,venv312,venv_rnn,__pycache__,.git,.github --max-line-length=100 --ignore=E501,W503,E402,F401,F841,F811,E302
format:
	black . --exclude 'venv|venv312|venv_rnn|__pycache__|.git|.github'
	isort experiments models framework --skip venv --skip venv312 --skip __pycache__ --skip .git --skip .github --skip data -v
tune:
	@echo "Running hyperparameter tuning with Optuna..."
	$(PYTHON) $(EXPERIMENTS_DIR)/train_with_optuna.py

train_final:
	@echo "Training final model with best Optuna hyperparameters..."
	$(PYTHON) $(EXPERIMENTS_DIR)/train_final_model.py

# Docker Commands

IMAGE_NAME = sms-spam-rnn
ECR_REPO = public.ecr.aws/k5v8x8j7/$(IMAGE_NAME)
REGION = us-east-1

docker-build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

docker-run:
	@echo " Running Docker container locally..."
	docker run --rm -it $(IMAGE_NAME)

docker-login:
	@echo " Logging in to AWS ECR..."
	aws ecr-public get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ECR_REPO)

docker-tag:
	@echo " Tagging Docker image..."
	docker tag $(IMAGE_NAME):latest $(ECR_REPO):latest

docker-push:
	@echo " Pushing Docker image to AWS ECR..."
	docker push $(ECR_REPO):latest

# Combo command
docker-all: docker-build docker-login docker-tag docker-push

clean:
	@echo "Cleaning temporary files..."
	-del /Q /F *.pyc 2>nul
	-del /Q /F *.pyo 2>nul
	-del /Q /F *.log 2>nul
	-del /Q /F *.pth 2>nul
	-del /Q /F *.json 2>nul
	-rmdir /S /Q __pycache__ 2>nul