# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install with longer timeout & retries
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# Copy the rest of your project
COPY . .

CMD ["python", "experiments/train_final_model.py"]