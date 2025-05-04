#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Module for "Do We Still Need Clinical Language Models?" Reproduction

This module implements the core training functionality for all model types and tasks.
"""

import os
import torch
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Tuple, Union

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from transformers import PreTrainedModel, get_linear_schedule_with_warmup
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for training."""
    model_name: str
    task: str
    data_fraction: Union[int, str]
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    patience: int = 3
    eval_steps: int = 100
    output_dir: str = "results"
    logging_steps: int = 20
    save_strategy: str = "epoch"  # "steps" or "epoch"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    seed: int = 42
    extra_args: Dict[str, Any] = field(default_factory=dict)


def train_model(
    model: PreTrainedModel,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    config: TrainingConfig,
    task_specific_fn: Optional[Callable] = None,
    scheduler=None
) -> Tuple[PreTrainedModel, Dict[str, Any]]:
    """
    Generic training loop with mixed precision support.

    Args:
        model: The model to train
        optimizer: Optimizer
        train_dataloader: Training data loader
        valid_dataloader: Validation data loader
        config: Training configuration
        task_specific_fn: Optional function for task-specific processing
        scheduler: Optional learning rate scheduler

    Returns:
        Trained model and training results
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Training on CPU")

    # Move model to device
    model = model.to(device)

    # Initialize mixed precision if requested
    scaler = GradScaler() if config.fp16 and torch.cuda.is_available() else None

    # Create scheduler if not provided
    if scheduler is None:
        total_steps = len(train_dataloader) * config.max_epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    # Training loop variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0
    train_losses = []
    eval_losses = []
    best_model = None
    train_start_time = time.time()

    logger.info(f"Starting training for {config.max_epochs} epochs")
    logger.info(f"Model: {config.model_name}, Task: {config.task}, Data Fraction: {config.data_fraction}")
    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0
        step_loss = 0
        epoch_start_time = time.time()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Apply task-specific processing if provided
            if task_specific_fn:
                batch = task_specific_fn(batch, model, training=True)

            # Forward pass with mixed precision if enabled
            if config.fp16 and torch.cuda.is_available():
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs.loss / config.gradient_accumulation_steps

            # Backward pass with mixed precision if enabled
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Track loss
            step_loss += loss.item() * config.gradient_accumulation_steps

            # Update weights on gradient accumulation boundary
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if config.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Update weights
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # Update learning rate
                scheduler.step()

                # Zero gradients
                optimizer.zero_grad()

                # Track global step
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({"loss": step_loss})
                epoch_loss += step_loss
                step_loss = 0

                # Periodic evaluation
                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    logger.info(f"Evaluating at step {global_step}...")
                    val_loss, val_metrics = evaluate_model(
                        model=model,
                        dataloader=valid_dataloader,
                        config=config,
                        task_specific_fn=task_specific_fn
                    )
                    model.train()  # Switch back to train mode
                    eval_losses.append((global_step, val_loss))

                    # Log validation metrics
                    logger.info(f"Step {global_step} | Val Loss: {val_loss:.4f} | "
                              f"Metrics: {val_metrics}")

                    # Check for improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0

                        # Save best model
                        if config.save_strategy == "steps":
                            model_save_path = os.path.join(
                                config.output_dir, 
                                f"step_{global_step}_val_loss_{val_loss:.4f}"
                            )
                            model.save_pretrained(model_save_path)
                            logger.info(f"Saved best model to {model_save_path}")
                            best_model = model_save_path
                    else:
                        logger.info(f"No improvement in validation loss")

                # Logging
                if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                    train_losses.append((global_step, step_loss))

        # End of epoch
        epoch_loss /= len(train_dataloader)
        epoch_time = time.time() - epoch_start_time

        logger.info(f"Epoch {epoch+1}/{config.max_epochs} completed in {epoch_time:.1f}s | "
                   f"Train Loss: {epoch_loss:.4f}")

        # Epoch-level evaluation
        logger.info(f"Evaluating at the end of epoch {epoch+1}...")
        val_loss, val_metrics = evaluate_model(
            model=model,
            dataloader=valid_dataloader,
            config=config,
            task_specific_fn=task_specific_fn
        )
        eval_losses.append((global_step, val_loss))

        # Log validation metrics
        logger.info(f"Epoch {epoch+1}/{config.max_epochs} | Val Loss: {val_loss:.4f} | "
                  f"Metrics: {val_metrics}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save best model
            if config.save_strategy == "epoch":
                model_save_path = os.path.join(
                    config.output_dir, 
                    f"epoch_{epoch+1}_val_loss_{val_loss:.4f}"
                )
                model.save_pretrained(model_save_path)
                logger.info(f"Saved best model to {model_save_path}")
                best_model = model_save_path
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epochs")

            # Early stopping
            if epochs_no_improve >= config.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # End of training
    train_time = time.time() - train_start_time
    logger.info(f"Training completed in {train_time:.1f}s")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # Load best model if saved
    if best_model is not None:
        logger.info(f"Loading best model from {best_model}")
        model = model.from_pretrained(best_model).to(device)

    # Return trained model and training results
    results = {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "training_time": train_time,
        "global_steps": global_step,
        "epochs_completed": epoch + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return model, results


def evaluate_model(
    model: PreTrainedModel,
    dataloader: DataLoader,
    config: TrainingConfig,
    task_specific_fn: Optional[Callable] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Generic evaluation loop.

    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader
        config: Training configuration
        task_specific_fn: Optional function for task-specific processing

    Returns:
        Average loss and dictionary of metrics
    """
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Apply task-specific processing if provided
            if task_specific_fn:
                batch, labels = task_specific_fn(batch, model, training=False, return_labels=True)
                all_labels.extend(labels.cpu().numpy())

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            # Get predictions
            if task_specific_fn:
                preds = task_specific_fn(batch, model, outputs=outputs, get_predictions=True)
                all_preds.extend(preds)

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    # Calculate metrics
    metrics = {}
    if task_specific_fn and all_labels and all_preds:
        metrics = task_specific_fn(None, model, labels=all_labels, predictions=all_preds, compute_metrics=True)

    return avg_loss, metrics


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Test config class
    config = TrainingConfig(
        model_name="t5-base",
        task="mednli",
        data_fraction=100,
        max_epochs=3
    )

    print(f"Training configuration: {config}")