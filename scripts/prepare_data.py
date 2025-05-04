#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Preparation Script for "Do We Still Need Clinical Language Models?" Reproduction

This script prepares the data for all three tasks (MedNLI, RadQA, CLIP) and creates
the necessary subsets for the limited data experiments.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mednli import create_mednli_subsets, analyze_mednli_dataset
from src.data.radqa import create_radqa_subsets, analyze_radqa_dataset
from src.data.clip import create_clip_subsets, analyze_clip_dataset
from src.data.utils import verify_dataset_integrity

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "data"
PERCENTAGES = [1, 5, 10, 25, 100]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare datasets for clinical LLMs study.'
    )

    # Task selection
    parser.add_argument('--tasks', nargs='+', choices=['mednli', 'radqa', 'clip', 'all'], 
                        default=['all'], help='Tasks to prepare data for')

    # Directory paths
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Directory containing original datasets')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save processed datasets')

    # Subset options
    parser.add_argument('--percentages', nargs='+', type=int, default=PERCENTAGES,
                        help='Percentages of data to create (e.g., 1 5 10 25 100)')

    # Analysis options
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze datasets without creating subsets')
    parser.add_argument('--validate', action='store_true',
                        help='Validate dataset integrity')

    return parser.parse_args()


def prepare_mednli(data_dir, output_dir, percentages, analyze_only=False):
    """Prepare MedNLI dataset."""
    logger.info("Processing MedNLI dataset")

    # Verify path to original files
    mednli_dir = os.path.join(data_dir, "mednli")
    if not os.path.exists(mednli_dir):
        logger.error(f"MedNLI directory not found: {mednli_dir}")
        return False

    train_file = os.path.join(mednli_dir, "mli_train_v1.jsonl")
    dev_file = os.path.join(mednli_dir, "mli_dev_v1.jsonl")
    test_file = os.path.join(mednli_dir, "mli_test_v1.jsonl")

    if not all(os.path.exists(f) for f in [train_file, dev_file, test_file]):
        logger.error("MedNLI files not found. Please ensure the original files are available.")
        return False

    # Create output directory
    mednli_output_dir = os.path.join(output_dir, "mednli")
    os.makedirs(mednli_output_dir, exist_ok=True)

    # Create full subset with original data
    full_dir = os.path.join(mednli_output_dir, "full")
    os.makedirs(full_dir, exist_ok=True)

    # Copy original files to full subset
    if not analyze_only:
        logger.info("Copying original files to full subset")
        for src_file, dst_name in [(train_file, "train.jsonl"), (dev_file, "dev.jsonl"), (test_file, "test.jsonl")]:
            dst_file = os.path.join(full_dir, dst_name)
            if not os.path.exists(dst_file):
                with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                    dst.write(src.read())

    # Create subsets for different percentages
    if not analyze_only:
        logger.info("Creating MedNLI subsets")
        create_mednli_subsets(mednli_dir, mednli_output_dir, percentages)

    # Analyze dataset
    logger.info("Analyzing MedNLI dataset")
    stats = analyze_mednli_dataset(output_dir)

    # Save stats
    stats_file = os.path.join(mednli_output_dir, "stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"MedNLI processing completed. Statistics saved to {stats_file}")
    return True


def prepare_radqa(data_dir, output_dir, percentages, analyze_only=False):
    """Prepare RadQA dataset."""
    logger.info("Processing RadQA dataset")

    # Verify path to original files
    radqa_dir = os.path.join(data_dir, "radqa")
    if not os.path.exists(radqa_dir):
        logger.error(f"RadQA directory not found: {radqa_dir}")
        return False

    train_file = os.path.join(radqa_dir, "train.json")
    dev_file = os.path.join(radqa_dir, "dev.json")
    test_file = os.path.join(radqa_dir, "test.json")

    if not all(os.path.exists(f) for f in [train_file, dev_file, test_file]):
        logger.error("RadQA files not found. Please ensure the original files are available.")
        return False

    # Create output directory
    radqa_output_dir = os.path.join(output_dir, "radqa")
    os.makedirs(radqa_output_dir, exist_ok=True)

    # Create full subset with original data
    full_dir = os.path.join(radqa_output_dir, "full")
    os.makedirs(full_dir, exist_ok=True)

    # Copy original files to full subset
    if not analyze_only:
        logger.info("Copying original files to full subset")
        for src_file, dst_name in [(train_file, "train.json"), (dev_file, "dev.json"), (test_file, "test.json")]:
            dst_file = os.path.join(full_dir, dst_name)
            if not os.path.exists(dst_file):
                with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                    dst.write(src.read())

    # Create subsets for different percentages
    if not analyze_only:
        logger.info("Creating RadQA subsets")
        create_radqa_subsets(radqa_dir, radqa_output_dir, percentages)

    # Analyze dataset
    logger.info("Analyzing RadQA dataset")
    stats = analyze_radqa_dataset(output_dir)

    # Save stats
    stats_file = os.path.join(radqa_output_dir, "stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"RadQA processing completed. Statistics saved to {stats_file}")
    return True


def prepare_clip(data_dir, output_dir, percentages, analyze_only=False):
    """Prepare CLIP dataset."""
    logger.info("Processing CLIP dataset")

    # Verify path to original files
    clip_dir = os.path.join(data_dir, "clip")
    if not os.path.exists(clip_dir):
        logger.error(f"CLIP directory not found: {clip_dir}")
        return False

    sentence_file = os.path.join(clip_dir, "sentence_level.csv")
    train_ids_file = os.path.join(clip_dir, "train_ids.csv")
    val_ids_file = os.path.join(clip_dir, "val_ids.csv")
    test_ids_file = os.path.join(clip_dir, "test_ids.csv")

    if not all(os.path.exists(f) for f in [sentence_file, train_ids_file, val_ids_file, test_ids_file]):
        logger.error("CLIP files not found. Please ensure the original files are available.")
        return False

    # Create output directory
    clip_output_dir = os.path.join(output_dir, "clip")
    os.makedirs(clip_output_dir, exist_ok=True)

    # Create full subset with original data
    full_dir = os.path.join(clip_output_dir, "full")
    os.makedirs(full_dir, exist_ok=True)

    # Copy original files to full subset
    if not analyze_only:
        logger.info("Copying original files to full subset")
        for src_file, dst_name in [
            (sentence_file, "sentence_level.csv"),
            (train_ids_file, "train_ids.csv"),
            (val_ids_file, "val_ids.csv"),
            (test_ids_file, "test_ids.csv")
        ]:
            dst_file = os.path.join(full_dir, dst_name)
            if not os.path.exists(dst_file):
                with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                    dst.write(src.read())

    # Create subsets for different percentages
    if not analyze_only:
        logger.info("Creating CLIP subsets")
        create_clip_subsets(clip_dir, clip_output_dir, percentages)

    # Analyze dataset
    logger.info("Analyzing CLIP dataset")
    stats = analyze_clip_dataset(output_dir)

    # Save stats
    stats_file = os.path.join(clip_output_dir, "stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"CLIP processing completed. Statistics saved to {stats_file}")
    return True


def main():
    """Main entrypoint."""
    args = parse_args()

    logger.info("Starting data preparation")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Percentages: {args.percentages}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate dataset integrity if requested
    if args.validate:
        logger.info("Validating dataset integrity")
        integrity_results = verify_dataset_integrity(args.data_dir)

        for task, result in integrity_results.items():
            if result['valid']:
                logger.info(f"✅ {task} dataset is valid")
            else:
                logger.error(f"❌ {task} dataset has issues: {result['issues']}")

    # Determine which tasks to process
    tasks_to_process = args.tasks
    if 'all' in tasks_to_process:
        tasks_to_process = ['mednli', 'radqa', 'clip']

    # Process each task
    results = {}

    if 'mednli' in tasks_to_process:
        results['mednli'] = prepare_mednli(args.data_dir, args.output_dir, args.percentages, args.analyze_only)

    if 'radqa' in tasks_to_process:
        results['radqa'] = prepare_radqa(args.data_dir, args.output_dir, args.percentages, args.analyze_only)

    if 'clip' in tasks_to_process:
        results['clip'] = prepare_clip(args.data_dir, args.output_dir, args.percentages, args.analyze_only)

    # Log summary
    logger.info("Data preparation summary:")
    for task, success in results.items():
        logger.info(f"{task}: {'✅ Success' if success else '❌ Failed'}")

    if all(results.values()):
        logger.info("All tasks completed successfully")
    else:
        logger.warning("Some tasks failed. Check the logs for details.")


if __name__ == "__main__":
    main()