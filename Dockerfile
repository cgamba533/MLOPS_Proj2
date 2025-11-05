# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model.py .
COPY data.py .
COPY train.py .

# Create directories for outputs
RUN mkdir -p /app/checkpoints /app/wandb

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - can be overridden
# Using your best hyperparameters from Project 1 (Optuna best)
CMD ["python", "train.py", \
     "--lr", "8e-5", \
     "--seq_classif_dropout", "0.586", \
     "--warmup_steps", "300", \
     "--checkpoint_dir", "/app/checkpoints"]