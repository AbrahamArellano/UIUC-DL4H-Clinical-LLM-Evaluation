#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Utilities Module

This module provides shared utilities for data processing across all tasks.
"""

import os
import json
import hashlib
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


def verify_dataset_integrity(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Verify the integrity of all datasets by checking file existence and format.

    Args:
        data_dir: Base data directory

    Returns:
        Dictionary of validation results for each task
    """
    integrity_results = {}

    # Verify MedNLI
    mednli_dir = os.path.join(data_dir, "mednli")
    train_file = os.path.join(mednli_dir, "mli_train_v1.jsonl")
    dev_file = os.path.join(mednli_dir, "mli_dev_v1.jsonl")
    test_file = os.path.join(mednli_dir, "mli_test_v1.jsonl")

    mednli_valid = True
    mednli_issues = []

    # Check file existence
    for file_path, file_name in [(train_file, "mli_train_v1.jsonl"), 
                              (dev_file, "mli_dev_v1.jsonl"),
                              (test_file, "mli_test_v1.jsonl")]:
        if not os.path.exists(file_path):
            mednli_valid = False
            mednli_issues.append(f"Missing file: {file_name}")

    # Check file format if all files exist
    if mednli_valid:
        try:
            with open(train_file, 'r') as f:
                sample_line = f.readline()
                sample_data = json.loads(sample_line)

                # Check required fields
                required_fields = ['gold_label', 'sentence1', 'sentence2']
                for field in required_fields:
                    if field not in sample_data:
                        mednli_valid = False
                        mednli_issues.append(f"Missing required field: {field}")

            # Verify label values
            valid_labels = {"entailment", "neutral", "contradiction"}
            with open(train_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # Check only first 100 examples
                        break
                    data = json.loads(line)
                    if data['gold_label'] not in valid_labels:
                        mednli_valid = False
                        mednli_issues.append(f"Invalid label: {data['gold_label']}")
                        break
        except json.JSONDecodeError:
            mednli_valid = False
            mednli_issues.append("Invalid JSON format")
        except Exception as e:
            mednli_valid = False
            mednli_issues.append(f"Error validating format: {str(e)}")

    integrity_results['mednli'] = {
        'valid': mednli_valid,
        'issues': mednli_issues
    }

    # Verify RadQA
    radqa_dir = os.path.join(data_dir, "radqa")
    train_file = os.path.join(radqa_dir, "train.json")
    dev_file = os.path.join(radqa_dir, "dev.json")
    test_file = os.path.join(radqa_dir, "test.json")

    radqa_valid = True
    radqa_issues = []

    # Check file existence
    for file_path, file_name in [(train_file, "train.json"), 
                              (dev_file, "dev.json"),
                              (test_file, "test.json")]:
        if not os.path.exists(file_path):
            radqa_valid = False
            radqa_issues.append(f"Missing file: {file_name}")

    # Check file format if all files exist
    if radqa_valid:
        try:
            with open(train_file, 'r') as f:
                data = json.load(f)

                # Check SQuAD format
                if 'data' not in data:
                    radqa_valid = False
                    radqa_issues.append("Missing 'data' field (SQuAD format)")
                elif not data['data']:
                    radqa_valid = False
                    radqa_issues.append("Empty 'data' field")
                elif 'paragraphs' not in data['data'][0]:
                    radqa_valid = False
                    radqa_issues.append("Missing 'paragraphs' field in data[0]")
                else:
                    # Check sample paragraph
                    paragraph = data['data'][0]['paragraphs'][0]
                    if 'context' not in paragraph:
                        radqa_valid = False
                        radqa_issues.append("Missing 'context' field in paragraph")
                    if 'qas' not in paragraph:
                        radqa_valid = False
                        radqa_issues.append("Missing 'qas' field in paragraph")
                    elif not paragraph['qas']:
                        radqa_valid = False
                        radqa_issues.append("Empty 'qas' field")
                    else:
                        # Check sample QA
                        qa = paragraph['qas'][0]
                        required_qa_fields = ['id', 'question']
                        for field in required_qa_fields:
                            if field not in qa:
                                radqa_valid = False
                                radqa_issues.append(f"Missing required field: {field} in QA")
        except json.JSONDecodeError:
            radqa_valid = False
            radqa_issues.append("Invalid JSON format")
        except Exception as e:
            radqa_valid = False
            radqa_issues.append(f"Error validating format: {str(e)}")

    integrity_results['radqa'] = {
        'valid': radqa_valid,
        'issues': radqa_issues
    }

    # Verify CLIP
    clip_dir = os.path.join(data_dir, "clip")
    sentence_file = os.path.join(clip_dir, "sentence_level.csv")
    train_ids_file = os.path.join(clip_dir, "train_ids.csv")
    val_ids_file = os.path.join(clip_dir, "val_ids.csv")
    test_ids_file = os.path.join(clip_dir, "test_ids.csv")

    clip_valid = True
    clip_issues = []

    # Check file existence
    for file_path, file_name in [(sentence_file, "sentence_level.csv"), 
                              (train_ids_file, "train_ids.csv"),
                              (val_ids_file, "val_ids.csv"),
                              (test_ids_file, "test_ids.csv")]:
        if not os.path.exists(file_path):
            clip_valid = False
            clip_issues.append(f"Missing file: {file_name}")

    # Check file format if all files exist
    if clip_valid:
        try:
            # Check sentence file
            sentences_df = pd.read_csv(sentence_file)

            # Find ID column
            id_col = None
            for col in sentences_df.columns:
                if 'id' in col.lower():
                    id_col = col
                    break

            if id_col is None:
                clip_valid = False
                clip_issues.append("No ID column found in sentence_level.csv")

            if 'sentence' not in sentences_df.columns:
                clip_valid = False
                clip_issues.append("Missing 'sentence' column in sentence_level.csv")

            if 'labels' not in sentences_df.columns:
                clip_valid = False
                clip_issues.append("Missing 'labels' column in sentence_level.csv")

            # Check ID files
            for id_file, file_name in [(train_ids_file, "train_ids.csv"),
                                    (val_ids_file, "val_ids.csv"),
                                    (test_ids_file, "test_ids.csv")]:
                # Try to read as single-column CSV first
                try:
                    ids_df = pd.read_csv(id_file, header=None)
                    if ids_df.empty:
                        clip_valid = False
                        clip_issues.append(f"Empty file: {file_name}")
                except Exception:
                    # If failed, try with header
                    try:
                        ids_df = pd.read_csv(id_file)
                        if ids_df.empty:
                            clip_valid = False
                            clip_issues.append(f"Empty file: {file_name}")
                    except Exception as e:
                        clip_valid = False
                        clip_issues.append(f"Error reading {file_name}: {str(e)}")

        except Exception as e:
            clip_valid = False
            clip_issues.append(f"Error validating format: {str(e)}")

    integrity_results['clip'] = {
        'valid': clip_valid,
        'issues': clip_issues
    }

    return integrity_results


def compute_file_hash(file_path: str) -> str:
    """
    Compute the SHA-256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        SHA-256 hash hex string
    """
    sha = hashlib.sha256()

    with open(file_path, 'rb') as f:
        while True:
            block = f.read(65536)  # Read in 64K chunks
            if not block:
                break
            sha.update(block)

    return sha.hexdigest()


def get_task_paths(task: str, data_fraction: Union[int, str]) -> Dict[str, str]:
    """
    Get paths to data files for a specific task and data fraction.

    Args:
        task: Task name ('mednli', 'radqa', 'clip')
        data_fraction: Data fraction (1, 5, 10, 25, 100 or 'full', '1pct', etc.)

    Returns:
        Dictionary of file paths for the task
    """
    # Normalize data_fraction
    if isinstance(data_fraction, int):
        if data_fraction == 100:
            subset_dir = "full"
        else:
            subset_dir = f"{data_fraction}pct"
    else:
        subset_dir = data_fraction

    # Base data directory
    data_dir = os.path.join("data", task, subset_dir)

    # Task-specific paths
    if task == 'mednli':
        return {
            'train_file': os.path.join(data_dir, "train.jsonl"),
            'dev_file': os.path.join(data_dir, "dev.jsonl"),
            'test_file': os.path.join(data_dir, "test.jsonl")
        }
    elif task == 'radqa':
        return {
            'train_file': os.path.join(data_dir, "train.json"),
            'dev_file': os.path.join(data_dir, "dev.json"),
            'test_file': os.path.join(data_dir, "test.json")
        }
    elif task == 'clip':
        return {
            'sentence_file': os.path.join(data_dir, "sentence_level.csv"),
            'train_ids_file': os.path.join(data_dir, "train_ids.csv"),
            'val_ids_file': os.path.join(data_dir, "val_ids.csv"),
            'test_ids_file': os.path.join(data_dir, "test_ids.csv")
        }
    else:
        raise ValueError(f"Unknown task: {task}")


def create_directory_structure(base_dir: str) -> None:
    """
    Create the basic directory structure for the project.

    Args:
        base_dir: Base project directory
    """
    # Main directories
    directories = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "logs")
    ]

    # Task and subset directories
    for task in ['mednli', 'radqa', 'clip']:
        task_dir = os.path.join(base_dir, "data", task)
        directories.append(task_dir)

        for subset in ['full', '25pct', '10pct', '5pct', '1pct']:
            directories.append(os.path.join(task_dir, subset))

    # Model directories
    for model_type in ['pretrained', 'finetuned']:
        model_dir = os.path.join(base_dir, "models", model_type)
        directories.append(model_dir)

    # Results directories
    for task in ['mednli', 'radqa', 'clip']:
        directories.append(os.path.join(base_dir, "results", task))

    # Create all directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Test directory creation
    create_directory_structure("test_project")

    # Test task paths
    for task in ['mednli', 'radqa', 'clip']:
        for fraction in [1, 5, 10, 25, 100]:
            paths = get_task_paths(task, fraction)
            print(f"{task} @ {fraction}%: {paths}")