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
	flake8 . --exclude=venv,venv312,venv_rnn,__pycache__,.git,.github --max-line-length=100 --ignore=E501,W503
format:
	black . --exclude 'venv|venv312|__pycache__|.git|.github|\.gitignore|.*\.png|.*\.jpg|.*\.pkl' -v
	isort experiments models framework --skip venv --skip venv312 --skip __pycache__ --skip .git --skip .github --skip data -v
tune:
	@echo "Running hyperparameter tuning with Optuna..."
	$(PYTHON) $(EXPERIMENTS_DIR)/train_with_optuna.py

train_final:
	@echo "Training final model with best Optuna hyperparameters..."
	$(PYTHON) $(EXPERIMENTS_DIR)/train_final_model.py
# Authenticate Docker to AWS ECR Public
login:
	@echo "Logging in to AWS ECR Public..."
	aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(ECR_REGISTRY)

# Build Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

# Tag Docker image
tag:
	@echo "Tagging Docker image..."
	docker tag $(IMAGE_NAME):$(TAG) $(ECR_REGISTRY)/$(IMAGE_NAME):$(TAG)

# Push Docker image to ECR
push: login tag
	@echo "Pushing Docker image to ECR..."
	docker push $(ECR_REGISTRY)/$(IMAGE_NAME):$(TAG)

# Combined target: build, tag, and push in one go
deploy: build push

clean:
	@echo "Cleaning temporary files..."
	-del /Q /F *.pyc 2>nul
	-del /Q /F *.pyo 2>nul
	-del /Q /F *.log 2>nul
	-del /Q /F *.pth 2>nul
	-del /Q /F *.json 2>nul
	-rmdir /S /Q __pycache__ 2>nul