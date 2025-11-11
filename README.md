# Containerization of Fine-Tuned DistilBert Model

MLOps Project 2: Containerized training pipeline for fine-tuning DistilBERT on the MRPC paraphrase detection task from the GLUE benchmark.

## Overview

This project provides a containerized environment for fine-tuning DistilBERT on **MRPC** (Microsoft Research Paraphrase Corpus), a paraphrase detection task from the GLUE benchmark. The training pipeline is packaged in Docker for reproducibility and ease of deployment.

**Best Model Performance (from Project 1):**
- F1 Score: 0.9033
- Accuracy: 0.8652
- Hyperparameters: LR=8e-5, Dropout=0.586, Warmup Steps=300

## Task: MRPC Paraphrase Detection

**Dataset**: Microsoft Research Paraphrase Corpus (MRPC) from the GLUE benchmark

**Task**: Binary classification to determine if two sentences are semantic paraphrases

**Dataset Size**:
- Training: 3,668 sentence pairs
- Validation: 408 sentence pairs

**Example**:
- Sentence 1: "The bird is bathing in the sink"
- Sentence 2: "The bird is washing itself in the water basin"
- Label: 1 (paraphrase)

**Metrics**: F1 Score and Accuracy

## Quick Start Guide

### Prerequisites

- Docker installed
- (Optional) Weights & Biases account for experiment tracking

### Build and Run
```bash
# Clone the project repository
git clone https://github.com/cgamba533/MLOPS_Proj2.git
cd MLOPS_Proj2

# Build the Docker image (CPU-optimized - Recommended)
docker build -f Dockerfile.cpu -t glue-trainer-cpu .

# Alternative Docker image (Requires substantially more disk-space)
docker build -f Dockerfile -t glue-trainer

# Run training (without W&B Tracking)
docker run --rm glue-trainer-cpu --no_wandb

# Run training (with W&B Tracking)
docker run --rm -e WANDB_API_KEY=your_key_here glue-trainer-cpu
```

## Project File Structure
```

|_ train.py              # Main training script with CLI arguments
|_ model.py              # GLUETransformer Lightning module (Directly from Project 1)
|_ data.py               # Data loading module for GLUE tasks (Directly from Project 1)
|_ requirements.txt      # Python dependencies
|_ Dockerfile            # Standard Docker image (Uses full PyTorch)
|_ Dockerfile.cpu        # Optimized CPU-only image (recommended, installs only PyTorch for CPU)
|_ .dockerignore         # Docker build exclusions
|_ .gitignore            # Git exclusions
|_ README.md             # This file

```

## Using Docker Image Once Built

### Basic Training

Run with default hyperparameters (optimized from Project 1):
```bash
docker run --rm glue-trainer-cpu --no_wandb
```

### Custom Hyperparameters

Override default settings:
```bash
docker run --rm glue-trainer-cpu \
  python train.py \
  --lr 3e-5 \
  --warmup_steps 250 \
  --seq_classif_dropout 0.2 \
  --max_epochs 3 \
  --no_wandb
```

## Available Arguments

Can modify for further fine-tuning or adjustments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 2e-5 | Learning rate |
| `--warmup_steps` | 0 | Number of warmup steps |
| `--weight_decay` | 0.0 | Weight decay for regularization |
| `--train_batch_size` | 32 | Training batch size |
| `--seq_classif_dropout` | 0.2 | Dropout rate |
| `--lr_scheduler_type` | linear | Scheduler: linear, cosine, onecycle |
| `--max_epochs` | 3 | Number of training epochs |
| `--task_name` | mrpc | GLUE task (mrpc, cola, sst2, etc.) |
| `--checkpoint_dir` | checkpoints | Directory for saving models |
| `--no_wandb` | False | Disable W&B logging |

## Running on Cloud Platforms

### GitHub Codespaces (Can alternatively use Docker Playground)

1. Fork this repository
2. Click "Code" -> "Codespaces" -> "Create codespace on main"
3. In the terminal:
```bash
docker build -f Dockerfile.cpu -t glue-trainer-cpu .
docker run --rm -e WANDB_API_KEY=your_key_here glue-trainer-cpu
```

**Note**: Use `Dockerfile.cpu` to avoid disk space issues.

## Project Background

This is Project 2 for the MLOps course, building on hyperparameter tuning work from Project 1. The goal is to containerize a machine learning training pipeline for reproducibility and deployment.

**Key Learnings:**
- Dockerizing ML workflows
- Managing dependencies across environments
- Handling GPU vs CPU configurations
- Optimizing image sizes for different platforms
- Converting Jupyter notebooks to production-ready scripts

**Project 1 Context**: 
- Manual hyperparameter exploration
- Identified key parameters: learning rate, warmup steps, dropout
- Automated tuning with Optuna
- Best model achieved 0.9033 F1 score
