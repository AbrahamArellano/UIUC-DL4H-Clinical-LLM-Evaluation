#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MedNLI Dataset Module

This module provides dataset classes and utilities for the MedNLI task.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Constants
MEDNLI_LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
MEDNLI_ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}


class MedNLIDataset(Dataset):
    """Base MedNLI dataset class."""

    def __init__(self, jsonl_file: str, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        """
        Initialize MedNLI dataset.

        Args:
            jsonl_file: Path to JSONL file with MedNLI data
            tokenizer: HuggingFace tokenizer for encoding text
            max_length: Maximum sequence length for tokenization
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.examples = []

        # Load data from JSONL file
        with open(jsonl_file, 'r') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)

        # Map label to id
        self.label_map = MEDNLI_LABEL2ID

        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_file}")

    def __len__(self):
        return len(self.examples)

    def get_label_distribution(self):
        """Return label distribution in dataset."""
        labels = [example['gold_label'] for example in self.examples]
        return {label: labels.count(label) for label in set(labels)}


class MedNLIEncoderOnlyDataset(MedNLIDataset):
    """MedNLI dataset for encoder-only models (RoBERTa, BioClinicalBERT, etc.)"""

    def __init__(self, jsonl_file: str, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        super().__init__(jsonl_file, tokenizer, max_length)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        premise = example['sentence1']
        hypothesis = example['sentence2']
        label = self.label_map[example['gold_label']]

        # Format for encoder-only models: "{premise} [SEP] {hypothesis}"
        # The tokenizer will handle adding the special tokens
        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension from tokenizer output
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Add label
        item["labels"] = torch.tensor(label, dtype=torch.long)

        return item


class MedNLIEncoderDecoderDataset(MedNLIDataset):
    """MedNLI dataset for encoder-decoder models (T5, Clinical-T5, etc.)"""

    def __init__(self, jsonl_file: str, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        super().__init__(jsonl_file, tokenizer, max_length)
        # Map labels to text for T5
        self.id_to_label = MEDNLI_ID2LABEL

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        premise = example['sentence1']
        hypothesis = example['sentence2']
        label = self.label_map[example['gold_label']]

        # Format for T5: "mnli premise: {premise} hypothesis: {hypothesis}"
        input_text = f"mnli premise: {premise} hypothesis: {hypothesis}"
        target_text = self.id_to_label[label]

        # Encode input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Encode target
        target_encoding = self.tokenizer(
            target_text,
            max_length=8,  # Target is just a single word
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension
        item = {
            "input_ids": input_encoding['input_ids'].squeeze(0),
            "attention_mask": input_encoding['attention_mask'].squeeze(0),
            "labels": target_encoding['input_ids'].squeeze(0),
            "label_id": torch.tensor(label, dtype=torch.long)  # Original label ID for easy evaluation
        }

        # Replace padding tokens in labels with -100 (ignored in loss)
        item["labels"][item["labels"] == self.tokenizer.pad_token_id] = -100

        return item


def load_mednli_data(data_dir: str, data_fraction: int = 100) -> Tuple[str, str, str]:
    """
    Load MedNLI data files for the specified data fraction.

    Args:
        data_dir: Base data directory
        data_fraction: Percentage of training data to use (1, 5, 10, 25, 100)

    Returns:
        train_file, dev_file, test_file: Paths to the data files
    """
    # Determine appropriate data subdirectory based on fraction
    if data_fraction == 100:
        subset_dir = "full"
    else:
        subset_dir = f"{data_fraction}pct"

    # Get file paths
    mednli_dir = os.path.join(data_dir, "mednli", subset_dir)
    train_file = os.path.join(mednli_dir, "train.jsonl")
    dev_file = os.path.join(mednli_dir, "dev.jsonl")
    test_file = os.path.join(mednli_dir, "test.jsonl")

    # Verify files exist
    for file_path in [train_file, dev_file, test_file]:
        if not os.path.exists(file_path):
            logger.warning(f"Data file not found: {file_path}")

    return train_file, dev_file, test_file


def create_mednli_subsets(source_dir: str, output_dir: str, percentages: List[int] = [1, 5, 10, 25, 100]):
    """
    Create stratified subsets of MedNLI dataset.

    Args:
        source_dir: Directory containing original MedNLI data
        output_dir: Directory to save subsets
        percentages: List of percentages to create
    """
    logger.info("Creating MedNLI subsets")

    # Load training data
    train_file = os.path.join(source_dir, "mli_train_v1.jsonl")
    train_examples = []
    with open(train_file, 'r') as f:
        for line in f:
            train_examples.append(json.loads(line))

    # Convert to DataFrame for easier manipulation
    train_df = pd.DataFrame(train_examples)

    # Create directories and save subsets
    for percentage in percentages:
        if percentage == 100:
            subset_name = "full"
        else:
            subset_name = f"{percentage}pct"

        subset_dir = os.path.join(output_dir, subset_name)
        os.makedirs(subset_dir, exist_ok=True)

        # Create stratified subset
        if subset_name == 'full':
            subset_df = train_df
        else:
            # Stratified sampling to maintain label distribution
            subset_df = train_df.groupby('gold_label', group_keys=False).apply(
                lambda x: x.sample(frac=percentage/100, random_state=42)
            )

        # Save subset to file
        subset_file = os.path.join(subset_dir, "train.jsonl")
        with open(subset_file, 'w') as f:
            for _, row in subset_df.iterrows():
                f.write(json.dumps(row.to_dict()) + '\n')

        # Copy dev and test files
        dev_file = os.path.join(source_dir, "mli_dev_v1.jsonl")
        test_file = os.path.join(source_dir, "mli_test_v1.jsonl")

        for src_file, dst_name in [(dev_file, "dev.jsonl"), (test_file, "test.jsonl")]:
            dst_file = os.path.join(subset_dir, dst_name)
            with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                dst.write(src.read())

        # Log subset statistics
        label_counts = subset_df['gold_label'].value_counts().to_dict()
        logger.info(f"Created {subset_name} subset with {len(subset_df)} examples")
        logger.info(f"Label distribution: {label_counts}")


def prepare_mednli_dataloaders(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    data_fraction: str = 'full',
    batch_size: Optional[int] = None,
    max_length: int = 256,
    data_dir: str = 'data'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare DataLoaders for MedNLI task.

    Args:
        model_name: Name of the model to determine appropriate dataset class
        tokenizer: HuggingFace tokenizer
        data_fraction: Data subset to use ('full', '25pct', '10pct', '5pct', '1pct')
        batch_size: Batch size (will use model-appropriate default if None)
        max_length: Maximum sequence length
        data_dir: Base data directory

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    # Load data paths
    train_file, dev_file, test_file = load_mednli_data(data_dir, 
                                                     int(data_fraction.rstrip('pct')) if data_fraction != 'full' else 100)

    # Determine if we're using an encoder-decoder model
    is_encoder_decoder = any(name in model_name.lower() for name in ['t5', 'bart'])

    # Set batch size based on model type if not specified
        if batch_size is None:
            if 'large' in model_name.lower():
                batch_size = 16
            elif 'base' in model_name.lower():
                batch_size = 24
            else:
                batch_size = 32

        # Create appropriate datasets based on model type
        if is_encoder_decoder:
            train_dataset = MedNLIEncoderDecoderDataset(train_file, tokenizer, max_length)
            val_dataset = MedNLIEncoderDecoderDataset(dev_file, tokenizer, max_length)
            test_dataset = MedNLIEncoderDecoderDataset(test_file, tokenizer, max_length)
        else:
            train_dataset = MedNLIEncoderOnlyDataset(train_file, tokenizer, max_length)
            val_dataset = MedNLIEncoderOnlyDataset(dev_file, tokenizer, max_length)
            test_dataset = MedNLIEncoderOnlyDataset(test_file, tokenizer, max_length)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )

        logger.info(f"Prepared MedNLI dataloaders with batch size {batch_size}")
        logger.info(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples, Test: {len(test_dataset)} examples")

        return train_loader, val_loader, test_loader


    def analyze_mednli_dataset(data_dir: str) -> Dict:
        """
        Analyze MedNLI dataset and print statistics.

        Args:
            data_dir: Directory containing MedNLI files

        Returns:
            Dictionary of dataset statistics
        """
        logger.info("Analyzing MedNLI dataset")

        # Load data files
        train_file = os.path.join(data_dir, "mednli", "full", "train.jsonl")
        dev_file = os.path.join(data_dir, "mednli", "full", "dev.jsonl")
        test_file = os.path.join(data_dir, "mednli", "full", "test.jsonl")

        # Load examples
        train_examples = []
        dev_examples = []
        test_examples = []

        with open(train_file, 'r') as f:
            for line in f:
                train_examples.append(json.loads(line))

        with open(dev_file, 'r') as f:
            for line in f:
                dev_examples.append(json.loads(line))

        with open(test_file, 'r') as f:
            for line in f:
                test_examples.append(json.loads(line))

        # Count labels
        train_labels = [ex['gold_label'] for ex in train_examples]
        dev_labels = [ex['gold_label'] for ex in dev_examples]
        test_labels = [ex['gold_label'] for ex in test_examples]

        train_label_counts = {label: train_labels.count(label) for label in set(train_labels)}
        dev_label_counts = {label: dev_labels.count(label) for label in set(dev_labels)}
        test_label_counts = {label: test_labels.count(label) for label in set(test_labels)}

        # Calculate text lengths
        train_premise_lengths = [len(ex['sentence1'].split()) for ex in train_examples]
        train_hypothesis_lengths = [len(ex['sentence2'].split()) for ex in train_examples]

        # Calculate statistics
        stats = {
            'train_count': len(train_examples),
            'dev_count': len(dev_examples),
            'test_count': len(test_examples),
            'train_labels': train_label_counts,
            'dev_labels': dev_label_counts,
            'test_labels': test_label_counts,
            'avg_premise_length': np.mean(train_premise_lengths),
            'avg_hypothesis_length': np.mean(train_hypothesis_lengths),
            'max_premise_length': max(train_premise_lengths),
            'max_hypothesis_length': max(train_hypothesis_lengths)
        }

        # Log statistics
        logger.info(f"MedNLI dataset statistics:")
        logger.info(f"Train set: {stats['train_count']} examples")
        logger.info(f"Dev set: {stats['dev_count']} examples")
        logger.info(f"Test set: {stats['test_count']} examples")
        logger.info(f"Train label distribution: {stats['train_labels']}")
        logger.info(f"Average premise length: {stats['avg_premise_length']:.1f} words")
        logger.info(f"Average hypothesis length: {stats['avg_hypothesis_length']:.1f} words")

        return stats


    if __name__ == "__main__":
        # Set up logging for standalone execution
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Example usage
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        train_file = "data/mednli/full/train.jsonl"

        # Test dataset class
        dataset = MedNLIEncoderOnlyDataset(train_file, tokenizer)
        print(f"Dataset size: {len(dataset)}")
        print(f"Sample item: {dataset[0]}")
        print(f"Label distribution: {dataset.get_label_distribution()}")