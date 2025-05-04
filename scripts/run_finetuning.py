#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning experiment runner for "Do We Still Need Clinical Language Models?"

This script executes fine-tuning experiments across models, tasks, and data fractions.
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.model_registry import load_model, MODEL_REGISTRY
from src.training.trainer import train_model
from src.data.mednli import prepare_mednli_dataloaders
from src.data.radqa import prepare_radqa_dataloaders
from src.data.clip import prepare_clip_dataloaders
from src.training.task_functions import mednli_task_fn, radqa_task_fn, clip_task_fn
from src.evaluation.evaluator import evaluate_model
from src.training.config import TrainingConfig

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_RESULTS_DIR = "results"
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent
TASKS = ["mednli", "radqa", "clip"]
DATA_FRACTIONS = [1, 5, 10, 25, 100]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run fine-tuning experiments for clinical LLMs study.'
    )

    # Main options
    parser.add_argument('--model', type=str, help='Model name from model registry')
    parser.add_argument('--task', type=str, choices=TASKS, help='Task to run')
    parser.add_argument('--data_fraction', type=int, choices=DATA_FRACTIONS, 
                      help='Percentage of training data to use')

    # Batch options
    parser.add_argument('--all', action='store_true', help='Run all combinations')
    parser.add_argument('--all_models', action='store_true', help='Run all models for specified task and fraction')
    parser.add_argument('--all_tasks', action='store_true', help='Run all tasks for specified model and fraction')
    parser.add_argument('--all_fractions', action='store_true', help='Run all fractions for specified model and task')

    # Training options
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')

    # Output options
    parser.add_argument('--output_dir', type=str, default=DEFAULT_RESULTS_DIR, help='Output directory')
    parser.add_argument('--no_eval', action='store_true', help='Skip evaluation')

    args = parser.parse_args()

    # Validate args
    if args.all:
        pass  # No need to check other args
    elif (args.model is None or args.task is None or args.data_fraction is None) and not (
            args.all_models or args.all_tasks or args.all_fractions):
        parser.error("Must specify --all, or provide --model, --task, and --data_fraction, "
                   "or use --all_X arguments to run batches.")

    return args


def setup_output_dirs(output_base, task, model_name, data_fraction):
    """Set up output directories for experiment."""
    # Sanitize model name for directory
    model_dir = model_name.replace('/', '_')

    # Create output directory structure
    task_dir = os.path.join(output_base, task)
    model_dir = os.path.join(task_dir, model_dir)
    fraction_dir = os.path.join(model_dir, f"{data_fraction}pct")

    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fraction_dir, exist_ok=True)

    return fraction_dir


def run_experiment(model_name, task, data_fraction, args):
    """Run a single fine-tuning experiment."""
    logger.info(f"Starting experiment: {model_name} on {task} with {data_fraction}% data")

    start_time = time.time()

    # Setup output directory
    output_dir = setup_output_dirs(args.output_dir, task, model_name, data_fraction)

    # Check if this experiment has already been run
    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        logger.info(f"Experiment already completed. Results found at {results_file}")
        return

    # Load model and tokenizer
    try:
        model, tokenizer, model_config = load_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return

    # Get appropriate task function
    if task == "mednli":
        task_fn = mednli_task_fn
    elif task == "radqa":
        task_fn = radqa_task_fn
    elif task == "clip":
        task_fn = clip_task_fn
    else:
        logger.error(f"Unknown task: {task}")
        return

    # Determine data subset directory
    subset_dir = "full" if data_fraction == 100 else f"{data_fraction}pct"

    # Prepare data loaders
    try:
        if task == "mednli":
            train_loader, val_loader, test_loader = prepare_mednli_dataloaders(
                model_name, tokenizer, subset_dir, batch_size=args.batch_size
            )
        elif task == "radqa":
            train_loader, val_loader, test_loader = prepare_radqa_dataloaders(
                model_name, tokenizer, subset_dir, batch_size=args.batch_size
            )
        elif task == "clip":
            train_loader, val_loader, test_loader = prepare_clip_dataloaders(
                model_name, tokenizer, subset_dir, batch_size=args.batch_size
            )
    except Exception as e:
        logger.error(f"Failed to prepare data for {task} with {data_fraction}% data: {e}")
        return

    # Create training configuration
    config = TrainingConfig(
        model_name=model_name,
        task=task,
        data_fraction=data_fraction,
        batch_size=args.batch_size if args.batch_size else model_config.get("batch_size", 16),
        learning_rate=args.learning_rate if args.learning_rate else model_config.get("learning_rate", 2e-5),
        max_epochs=args.epochs if args.epochs else 5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=args.fp16,
        output_dir=output_dir
    )

    # Set up optimizer
    from torch.optim import AdamW
    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Train the model
    try:
        logger.info(f"Training {model_name} on {task} with {data_fraction}% data")
        trained_model, train_results = train_model(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            valid_dataloader=val_loader,
            config=config,
            task_specific_fn=task_fn
        )

        # Evaluate on test set if required
        if not args.no_eval:
            logger.info(f"Evaluating {model_name} on {task} test set")
            test_loss, test_metrics = evaluate_model(
                model=trained_model,
                dataloader=test_loader,
                config=config,
                task_specific_fn=task_fn
            )

            # Combine results
            results = {
                "model": model_name,
                "task": task,
                "data_fraction": data_fraction,
                "training_metrics": train_results,
                "test_metrics": test_metrics,
                "test_loss": float(test_loss),
                "time_seconds": time.time() - start_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            results = {
                "model": model_name,
                "task": task,
                "data_fraction": data_fraction,
                "training_metrics": train_results,
                "time_seconds": time.time() - start_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Experiment completed. Results saved to {results_file}")

        # Save results to unified task file
        task_results_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_results_dir, exist_ok=True)
        task_results_file = os.path.join(task_results_dir, 
                                       f"{model_name.replace('/', '_')}_{data_fraction}pct.json")

        with open(task_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results also saved to {task_results_file}")

        return results

    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        # Log failure
        with open(os.path.join(output_dir, "error.log"), 'w') as f:
            f.write(f"Error during experiment: {e}\n")
        return None


def run_experiments(args):
    """Run multiple experiments based on args."""
    # Determine which models to run
    if args.all or args.all_models:
        models = list(MODEL_REGISTRY.keys())
    elif args.model:
        if args.model not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {args.model}")
            return
        models = [args.model]
    else:
        logger.error("No model specified")
        return

    # Determine which tasks to run
    if args.all or args.all_tasks:
        tasks = TASKS
    elif args.task:
        tasks = [args.task]
    else:
        logger.error("No task specified")
        return

    # Determine which data fractions to run
    if args.all or args.all_fractions:
        fractions = DATA_FRACTIONS
    elif args.data_fraction:
        fractions = [args.data_fraction]
    else:
        logger.error("No data fraction specified")
        return

    # Run all specified experiments
    total_experiments = len(models) * len(tasks) * len(fractions)
    logger.info(f"Running {total_experiments} experiments")

    results = []
    for model in models:
        for task in tasks:
            for fraction in fractions:
                try:
                    result = run_experiment(model, task, fraction, args)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to run experiment {model} on {task} with {fraction}% data: {e}")

    return results


def main():
    """Main entrypoint."""
    args = parse_args()

    logger.info("Starting fine-tuning experiments")
    logger.info(f"Arguments: {args}")

    # Run experiments
    results = run_experiments(args)

    if results:
        logger.info(f"Successfully completed {len(results)} experiments")
    else:
        logger.warning("No experiments were successfully completed")


if __name__ == "__main__":
    main()