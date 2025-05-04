#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
In-Context Learning (ICL) experiment runner for "Do We Still Need Clinical Language Models?"

This script executes ICL experiments across models, tasks, and shot counts.
"""

import os
import sys
import json
import argparse
import logging
import time
import random
import traceback
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.icl import (
    MedNLIInContextLearner,
    RadQAInContextLearner,
    CLIPInContextLearner
)
from src.data.utils import get_task_paths

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_RESULTS_DIR = "results/icl"
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent
TASKS = ["mednli", "radqa", "clip"]
SHOTS = [1, 3, 5]
DATA_FRACTIONS = [1, 5, 10, 25, 100]
MODELS = {
    "t5-base": "google/flan-t5-base",
    "t5-large": "google/flan-t5-large", 
    "bioclinroberta": "emilyalsentzer/Bio_ClinicalBERT",
    "clinical-t5-base": "StanfordAIMI/clinical-t5-base",
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run in-context learning experiments for clinical LLMs study.'
    )

    # Main options
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()), 
                      help='Model name')
    parser.add_argument('--task', type=str, choices=TASKS, 
                      help='Task to run')
    parser.add_argument('--shots', type=int, choices=SHOTS, 
                      help='Number of few-shot examples')
    parser.add_argument('--data_fraction', type=int, choices=DATA_FRACTIONS, 
                      help='Percentage of training data to use for examples')

    # Batch options
    parser.add_argument('--all', action='store_true', help='Run all combinations')
    parser.add_argument('--all_models', action='store_true', help='Run all models for specified task')
    parser.add_argument('--all_tasks', action='store_true', help='Run all tasks for specified model')
    parser.add_argument('--all_shots', action='store_true', help='Run all shot counts')

    # Execution options
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (fewer samples)')
    parser.add_argument('--max_examples', type=int, default=None, 
                      help='Maximum number of examples to evaluate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Output options
    parser.add_argument('--output_dir', type=str, default=DEFAULT_RESULTS_DIR, help='Output directory')

    args = parser.parse_args()

    # Validate args
    if args.all:
        pass  # No need to check other args
    elif not any([args.all_models, args.all_tasks, args.all_shots]) and \
         (args.model is None or args.task is None or args.shots is None):
        parser.error("Must specify --all, or provide --model, --task, and --shots, "
                   "or use --all_X arguments to run batches.")

    # Set default data_fraction if not specified
    if args.data_fraction is None:
        args.data_fraction = 100

    return args


def setup_output_dirs(output_base, task):
    """Set up output directories for experiment."""
    # Create output directory structure
    task_dir = os.path.join(output_base, f"{task}_results")
    visualizations_dir = os.path.join(task_dir, "visualizations")

    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    return task_dir, visualizations_dir


def run_mednli_experiment(model_name, model_path, shots, data_fraction, args):
    """Run MedNLI ICL experiment."""
    task_dir, _ = setup_output_dirs(args.output_dir, "mednli")

    # Initialize model
    try:
        logger.info(f"Initializing {model_name} for MedNLI")
        learner = MedNLIInContextLearner(
            model_path=model_path,
            verbose=True,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        traceback.print_exc()
        return None

    # Get file paths
    paths = get_task_paths("mednli", data_fraction)

    # Result file path
    result_file = os.path.join(
        task_dir,
        f"mednli_{model_name.lower().replace('/', '_')}_{data_fraction}pct_{shots}shot_results.json"
    )

    # Skip if already exists
    if os.path.exists(result_file):
        logger.info(f"Results already exist at {result_file}, skipping")
        return None

    logger.info(f"Starting MedNLI evaluation with {shots} shots")

    try:
        # Run evaluation
        start_time = time.time()
        metrics = learner.evaluate_mednli(
            test_file=paths["test_file"],
            examples_file=paths["train_file"],
            num_samples=args.max_examples if args.debug else None,
            num_examples=shots
        )
        duration = time.time() - start_time

        # Add metadata
        metrics.update({
            "task": "mednli",
            "model": model_name,
            "data_fraction": data_fraction,
            "num_shots": shots,
            "runtime_seconds": duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Save detailed predictions if available
        if "predictions" in metrics:
            pred_file = os.path.join(
                task_dir,
                f"mednli_{model_name.lower().replace('/', '_')}_{data_fraction}pct_{shots}shot_predictions.json"
            )

            with open(pred_file, 'w') as f:
                json.dump(metrics["predictions"], f, indent=2)

            # Remove predictions from summary metrics
            metrics_summary = {k: v for k, v in metrics.items() if k != "predictions"}
            metrics_summary["predictions_file"] = pred_file
        else:
            metrics_summary = metrics

        # Save results
        with open(result_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        logger.info(f"Results saved to {result_file}")

        # Clean up
        learner.cleanup()

        return metrics_summary

    except Exception as e:
        logger.error(f"Error in MedNLI evaluation: {e}")
        traceback.print_exc()

        # Record error
        error_result = {
            "task": "mednli",
            "model": model_name,
            "data_fraction": data_fraction,
            "num_shots": shots,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(result_file, 'w') as f:
            json.dump(error_result, f, indent=2)

        return None


def run_radqa_experiment(model_name, model_path, shots, data_fraction, args):
    """Run RadQA ICL experiment."""
    task_dir, _ = setup_output_dirs(args.output_dir, "radqa")

    # Initialize model
    try:
        logger.info(f"Initializing {model_name} for RadQA")
        learner = RadQAInContextLearner(
            model_path=model_path,
            verbose=True,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        traceback.print_exc()
        return None

    # Get file paths
    paths = get_task_paths("radqa", data_fraction)

    # Result file path
    result_file = os.path.join(
        task_dir,
        f"radqa_{model_name.lower().replace('/', '_')}_{data_fraction}pct_{shots}shot_results.json"
    )

    # Skip if already exists
    if os.path.exists(result_file):
        logger.info(f"Results already exist at {result_file}, skipping")
        return None

    logger.info(f"Starting RadQA evaluation with {shots} shots")

    try:
        # Run evaluation
        start_time = time.time()
        metrics = learner.evaluate_radqa(
            test_file=paths["test_file"],
            examples_file=paths["train_file"],
            num_samples=args.max_examples if args.debug else None,
            num_examples=shots
        )
        duration = time.time() - start_time

        # Add metadata
        metrics.update({
            "task": "radqa",
            "model": model_name,
            "data_fraction": data_fraction,
            "num_shots": shots,
            "runtime_seconds": duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Save detailed predictions if available
        if "predictions" in metrics:
            pred_file = os.path.join(
                task_dir,
                f"radqa_{model_name.lower().replace('/', '_')}_{data_fraction}pct_{shots}shot_predictions.json"
            )

            with open(pred_file, 'w') as f:
                json.dump(metrics["predictions"], f, indent=2)

            # Remove predictions from summary metrics
            metrics_summary = {k: v for k, v in metrics.items() if k != "predictions"}
            metrics_summary["predictions_file"] = pred_file
        else:
            metrics_summary = metrics

        # Save results
        with open(result_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        logger.info(f"Results saved to {result_file}")

        # Clean up
        learner.cleanup()

        return metrics_summary

    except Exception as e:
        logger.error(f"Error in RadQA evaluation: {e}")
        traceback.print_exc()

        # Record error
        error_result = {
            "task": "radqa",
            "model": model_name,
            "data_fraction": data_fraction,
            "num_shots": shots,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(result_file, 'w') as f:
            json.dump(error_result, f, indent=2)

        return None


def run_clip_experiment(model_name, model_path, shots, data_fraction, args):
    """Run CLIP ICL experiment."""
    task_dir, _ = setup_output_dirs(args.output_dir, "clip")

    # Initialize model
    try:
        logger.info(f"Initializing {model_name} for CLIP")
        learner = CLIPInContextLearner(
            model_path=model_path,
            verbose=True,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        traceback.print_exc()
        return None

    # Get file paths
    paths = get_task_paths("clip", data_fraction)

    # Result file path
    result_file = os.path.join(
        task_dir,
        f"clip_{model_name.lower().replace('/', '_')}_{data_fraction}pct_{shots}shot_results.json"
    )

    # Skip if already exists
    if os.path.exists(result_file):
        logger.info(f"Results already exist at {result_file}, skipping")
        return None

    logger.info(f"Starting CLIP evaluation with {shots} shots")

    try:
        # Run evaluation
        start_time = time.time()
        metrics = learner.evaluate_clip(
            sentence_file=paths["sentence_file"],
            ids_file=paths["test_ids_file"],
            examples_file=paths["sentence_file"],
            examples_ids_file=paths["train_ids_file"],
            num_examples=shots,
            max_examples=args.max_examples if args.debug else None
        )
        duration = time.time() - start_time

        # Add metadata
        metrics.update({
            "task": "clip",
            "model": model_name,
            "data_fraction": data_fraction,
            "num_shots": shots,
            "runtime_seconds": duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Save detailed predictions if available
        if "predictions" in metrics:
            pred_file = os.path.join(
                task_dir,
                f"clip_{model_name.lower().replace('/', '_')}_{data_fraction}pct_{shots}shot_predictions.json"
            )

            with open(pred_file, 'w') as f:
                json.dump({
                    "predictions": metrics.get("predictions", []),
                    "ground_truths": metrics.get("ground_truths", []),
                    "label_names": ["appointment-related", "medication-related", "lab-related", 
                                   "patient-instructions", "procedure-related", "imaging-related", "other"]
                }, f, indent=2)

            # Remove large arrays from summary metrics
            metrics_summary = {k: v for k, v in metrics.items() 
                              if k not in ["predictions", "ground_truths"]}
            metrics_summary["predictions_file"] = pred_file
        else:
            metrics_summary = metrics

        # Save results
        with open(result_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        logger.info(f"Results saved to {result_file}")

        # Clean up
        learner.cleanup()

        return metrics_summary

    except Exception as e:
        logger.error(f"Error in CLIP evaluation: {e}")
        traceback.print_exc()

        # Record error
        error_result = {
            "task": "clip",
            "model": model_name,
            "data_fraction": data_fraction,
            "num_shots": shots,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(result_file, 'w') as f:
            json.dump(error_result, f, indent=2)

        return None


def run_experiment(model_name, task, shots, data_fraction, args):
    """Run a single ICL experiment."""
    # Get model path
    model_path = MODELS[model_name]

    logger.info(f"Running experiment: {model_name} on {task} with {shots} shots "
              f"and {data_fraction}% data")

    # Run task-specific experiment
    if task == "mednli":
        return run_mednli_experiment(model_name, model_path, shots, data_fraction, args)
    elif task == "radqa":
        return run_radqa_experiment(model_name, model_path, shots, data_fraction, args)
    elif task == "clip":
        return run_clip_experiment(model_name, model_path, shots, data_fraction, args)
    else:
        logger.error(f"Unknown task: {task}")
        return None


def run_experiments(args):
    """Run multiple experiments based on args."""
    # Set random seed
    random.seed(args.seed)

    # Determine which models to run
    if args.all or args.all_models:
        models = list(MODELS.keys())
    elif args.model:
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

    # Determine which shot counts to run
    if args.all or args.all_shots:
        shots_list = SHOTS
    elif args.shots:
        shots_list = [args.shots]
    else:
        logger.error("No shot count specified")
        return

    # Run all specified experiments
    total_experiments = len(models) * len(tasks) * len(shots_list)
    logger.info(f"Running {total_experiments} experiments")

    results = {}
    for model_name in models:
        results[model_name] = {}
        for task in tasks:
            for shots in shots_list:
                try:
                    # Log experiment start
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Running {task} with {model_name} using {shots} shots")
                    logger.info(f"{'='*80}\n")

                    result = run_experiment(model_name, task, shots, args.data_fraction, args)

                    # Save to all-results structure
                    if task not in results[model_name]:
                        results[model_name][task] = []
                    results[model_name][task].append(result)
                except Exception as e:
                    logger.error(f"Failed to run experiment {model_name} on {task} with {shots} shots: {e}")

    # Save complete results
    for task in tasks:
        task_dir, _ = setup_output_dirs(args.output_dir, task)
        complete_results_file = os.path.join(task_dir, f"{task}_icl_complete_results.json")

        # Extract only results for this task
        task_results = {}
        for model in results:
            task_results[model] = {}
            for frac, res_list in results[model].items():
                if frac == task:
                    task_results[model][str(args.data_fraction)] = res_list

        with open(complete_results_file, 'w') as f:
            json.dump(task_results, f, indent=2)

        logger.info(f"Saved complete results for {task} to: {complete_results_file}")

    return results


def main():
    """Main entrypoint."""
    args = parse_args()

    logger.info("Starting in-context learning experiments")
    logger.info(f"Arguments: {args}")

    # Run experiments
    results = run_experiments(args)

    if results:
        logger.info(f"Successfully completed experiments")
    else:
        logger.warning("No experiments were successfully completed")


if __name__ == "__main__":
    main()