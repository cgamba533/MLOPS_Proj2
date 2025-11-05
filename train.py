"""
Training script for GLUE fine-tuning with command-line arguments
"""

"""
Best performing model from Project 1: python train.py --lr 8e-5 --seq_classif_dropout 0.586 --warmup_steps 300
"""

import argparse
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model import GLUETransformer
from data import GLUEDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GLUE model with configurable hyperparameters")

    # Model and task configuration
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Pretrained model name or path")
    parser.add_argument("--task_name", type=str, default="mrpc",
                        choices=["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"],
                        help="GLUE task name")

    # Training hyperparameters
    parser.add_argument("--learning_rate", "--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for regularization")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--seq_classif_dropout", type=float, default=0.2,
                        help="Dropout rate for sequence classification head")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine", "polynomial", "constant", "onecycle"],
                        help="Learning rate scheduler type")
    parser.add_argument("--max_epochs", type=int, default=3,
                        help="Maximum number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length for tokenization")

    # Paths and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--wandb_project", type=str, default="mlops-proj2",
                        help="Weights & Biases project name")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Other options
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--offline", action="store_true",
                        help="Run W&B in offline mode")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    L.seed_everything(args.seed)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data module
    dm = GLUEDataModule(
        model_name_or_path=args.model_name,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")

    # Initialize model
    model = GLUETransformer(
        model_name_or_path=args.model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seq_classif_dropout=args.seq_classif_dropout,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    # Set up logger
    logger = None
    if not args.no_wandb:
        # Generate experiment name if not provided
        exp_name = args.experiment_name
        if exp_name is None:
            exp_name = f"lr{args.learning_rate}_warmup{args.warmup_steps}_dropout{args.seq_classif_dropout}"

        logger = WandbLogger(
            project=args.wandb_project,
            name=exp_name,
            offline=args.offline,
        )
        # Log hyperparameters
        logger.log_hyperparams(vars(args))

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{args.task_name}-{{epoch}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
    ]

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        deterministic=True,
    )

    # Print training configuration
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Warmup Steps: {args.warmup_steps}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Dropout: {args.seq_classif_dropout}")
    print(f"Batch Size: {args.train_batch_size}")
    print(f"LR Scheduler: {args.lr_scheduler_type}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print("=" * 60 + "\n")

    # Train the model
    trainer.fit(model, datamodule=dm)

    # Print final metrics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    final_metrics = trainer.callback_metrics
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")
    print("=" * 60 + "\n")

    # Save final model
    final_checkpoint = checkpoint_dir / f"{args.task_name}_final.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    print(f"Final model saved to: {final_checkpoint}")


if __name__ == "__main__":
    main()