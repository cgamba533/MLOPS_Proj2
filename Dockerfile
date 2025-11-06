FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .
COPY data.py .
COPY train.py .

RUN mkdir -p /app/checkpoints /app/wandb

ENV PYTHONUNBUFFERED=1

# Using best hyperparameters from Project 1 (Optuna best)
CMD ["python", "train.py", \
     "--lr", "8e-5", \
     "--seq_classif_dropout", "0.586", \
     "--warmup_steps", "300", \
     "--checkpoint_dir", "/app/checkpoints"]