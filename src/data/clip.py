#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLIP Dataset Module

This module provides dataset classes and utilities for the CLIP task
(Clinical Language Inference for Patient Follow-up Prediction).
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
CLIP_LABELS = [
    'appointment-related',
    'medication-related',
    'lab-related',
    'patient-instructions',
    'procedure-related',
    'imaging-related',
    'other'
]


class CLIPDataset(Dataset):
    """Base class for CLIP dataset for multi-label classification."""

    def __init__(self, 
                 sentence_file: str, 
                 ids_file: str, 
                 tokenizer: PreTrainedTokenizer, 
                 max_length: int = 256):
        """
        Initialize CLIP dataset.

        Args:
            sentence_file: Path to CSV file with sentence data
            ids_file: Path to CSV file with IDs for this split
            tokenizer: HuggingFace tokenizer for encoding text
            max_length: Maximum sequence length for tokenization
        """
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Load sentences data
        self.sentences_df = pd.read_csv(sentence_file)

        # Load IDs for this split - try with and without headers
        try:
            self.ids_df = pd.read_csv(ids_file)
        except:
            self.ids_df = pd.read_csv(ids_file, header=None, names=['doc_id'])

        # Find the document ID column in sentences_df
        self.doc_id_col = None
        for col in self.sentences_df.columns:
            if 'id' in col.lower():
                self.doc_id_col = col
                break

        if self.doc_id_col is None:
            raise ValueError("Could not find ID column in sentence file")

        # Filter sentences to include only IDs in this split
        note_ids = set(self.ids_df['doc_id'].values)
        self.filtered_sentences = self.sentences_df[self.sentences_df[self.doc_id_col].isin(note_ids)].copy()

        # Process labels
        # Check if labels are already in one-hot format
        missing_labels = [label for label in CLIP_LABELS if label not in self.filtered_sentences.columns]

        if 'labels' in self.filtered_sentences.columns:
            # Parse labels from string field
            for label in CLIP_LABELS:
                self.filtered_sentences[label] = self.filtered_sentences['labels'].str.lower().str.contains(
                    label.lower()).astype(int)
        elif missing_labels:
            # Need to create missing label columns
            for label in missing_labels:
                self.filtered_sentences[label] = 0

        logger.info(f"Loaded {len(self.filtered_sentences)} sentences for CLIP dataset")

    def __len__(self):
        return len(self.filtered_sentences)

    def get_label_distribution(self):
        """Return label distribution in dataset."""
        return {label: int(self.filtered_sentences[label].sum()) 
                for label in CLIP_LABELS if label in self.filtered_sentences.columns}


class CLIPEncoderOnlyDataset(CLIPDataset):
    """CLIP dataset for encoder-only models (RoBERTa, BioClinicalBERT, etc.)"""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.filtered_sentences.iloc[idx]
        text = row['sentence']

        # Create multi-hot label encoding
        labels = torch.zeros(len(CLIP_LABELS), dtype=torch.float)
        for i, label in enumerate(CLIP_LABELS):
            if label in row and row[label] == 1:
                labels[i] = 1.0

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension from tokenizer output
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Add label
        item["labels"] = labels
        item["doc_id"] = row[self.doc_id_col]

        return item


class CLIPEncoderDecoderDataset(CLIPDataset):
    """CLIP dataset for encoder-decoder models (T5, Clinical-T5, etc.)"""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.filtered_sentences.iloc[idx]
        text = row['sentence']

        # Format for T5: "clip: {sentence}"
        input_text = f"clip: {text}"

        # Create list of active labels
        active_labels = []
        for label in CLIP_LABELS:
            if label in row and row[label] == 1:
                active_labels.append(label)

        # Target is comma-separated list of labels
        target_text = ", ".join(active_labels) if active_labels else "none"

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=64,  # Target should be relatively short
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension
        item = {
            "input_ids": input_encoding['input_ids'].squeeze(0),
            "attention_mask": input_encoding['attention_mask'].squeeze(0),
            "labels": target_encoding['input_ids'].squeeze(0),
            "doc_id": row[self.doc_id_col]
        }

        # Replace padding tokens in labels with -100 (ignored in loss)
        item["labels"][item["labels"] == self.tokenizer.pad_token_id] = -100

        return item


def load_clip_data(data_dir: str, data_fraction: int = 100) -> Tuple[str, str, str, str]:
    """
    Load CLIP data files for the specified data fraction.

    Args:
        data_dir: Base data directory
        data_fraction: Percentage of training data to use (1, 5, 10, 25, 100)

    Returns:
        sentence_file, train_ids_file, val_ids_file, test_ids_file: Paths to the data files
    """
    # Determine appropriate data subdirectory based on fraction
    if data_fraction == 100:
        subset_dir = "full"
    else:
        subset_dir = f"{data_fraction}pct"

    # Get file paths
    clip_dir = os.path.join(data_dir, "clip", subset_dir)
    sentence_file = os.path.join(clip_dir, "sentence_level.csv")
    train_ids_file = os.path.join(clip_dir, "train_ids.csv")
    val_ids_file = os.path.join(clip_dir, "val_ids.csv")
    test_ids_file = os.path.join(clip_dir, "test_ids.csv")

    # Verify files exist
    for file_path in [sentence_file, train_ids_file, val_ids_file, test_ids_file]:
        if not os.path.exists(file_path):
            logger.warning(f"Data file not found: {file_path}")

    return sentence_file, train_ids_file, val_ids_file, test_ids_file


def create_clip_subsets(source_dir: str, output_dir: str, percentages: List[int] = [1, 5, 10, 25, 100]):
    """
    Create subsets of CLIP dataset by sampling note IDs.

    Args:
        source_dir: Directory containing original CLIP data
        output_dir: Directory to save subsets
        percentages: List of percentages to create
    """
    logger.info("Creating CLIP subsets")

    # Load original files
    sentence_file = os.path.join(source_dir, "sentence_level.csv")
    train_ids_file = os.path.join(source_dir, "train_ids.csv")
    val_ids_file = os.path.join(source_dir, "val_ids.csv")
    test_ids_file = os.path.join(source_dir, "test_ids.csv")

    # Check files exist
    if not all(os.path.exists(f) for f in [sentence_file, train_ids_file, val_ids_file, test_ids_file]):
        logger.error("CLIP files not found. Please ensure the original files are available.")
        return False

    # Load sentence data
    sentences_df = pd.read_csv(sentence_file)

    # Find ID column in sentences
    id_col = None
    for col in sentences_df.columns:
        if 'id' in col.lower():
            id_col = col
            break

    if id_col is None:
        logger.error("Could not find ID column in sentence file")
        return False

    # Load training IDs - try with and without headers
    try:
        train_ids_df = pd.read_csv(train_ids_file)
    except:
        train_ids_df = pd.read_csv(train_ids_file, header=None, names=['doc_id'])

    # Create directories for each subset
    for percentage in percentages:
        if percentage == 100:
            subset_name = "full"
        else:
            subset_name = f"{percentage}pct"

        subset_dir = os.path.join(output_dir, subset_name)
        os.makedirs(subset_dir, exist_ok=True)

        # Copy sentence file to subset directory
        subset_sentence_file = os.path.join(subset_dir, "sentence_level.csv")
        with open(sentence_file, 'r') as src, open(subset_sentence_file, 'w') as dst:
            dst.write(src.read())

        # For full dataset, copy original ID files
        if subset_name == 'full':
            for src_file, dst_name in [
                (train_ids_file, "train_ids.csv"),
                (val_ids_file, "val_ids.csv"),
                (test_ids_file, "test_ids.csv")
            ]:
                dst_file = os.path.join(subset_dir, dst_name)
                with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                    dst.write(src.read())
        else:
            # Sample a percentage of training IDs
            np.random.seed(42)  # For reproducibility
            sampled_ids = train_ids_df.sample(frac=percentage/100, random_state=42)

            # Save sampled IDs
            sampled_ids.to_csv(os.path.join(subset_dir, "train_ids.csv"), index=False)

            # Copy validation and test IDs unchanged
            for src_file, dst_name in [
                (val_ids_file, "val_ids.csv"),
                (test_ids_file, "test_ids.csv")
            ]:
                dst_file = os.path.join(subset_dir, dst_name)
                with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                    dst.write(src.read())

            # Log subset creation
            logger.info(f"Created {subset_name} subset with {len(sampled_ids)} note IDs")

    return True


def prepare_clip_dataloaders(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    data_fraction: str = 'full',
    batch_size: Optional[int] = None,
    max_length: int = 256,
    data_dir: str = 'data'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare DataLoaders for CLIP task.

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
    sentence_file, train_ids_file, val_ids_file, test_ids_file = load_clip_data(
        data_dir, 
        int(data_fraction.rstrip('pct')) if data_fraction != 'full' else 100
    )

    # Determine if we're using an encoder-decoder model
    is_encoder_decoder = any(name in model_name.lower() for name in ['t5', 'bart'])

    # Set batch size based on model type if not specified
    if batch_size is None:
        if 'large' in model_name.lower():
            batch_size = 16  # Larger models need smaller batch size
        elif 'base' in model_name.lower():
            batch_size = 24
        else:
            batch_size = 32

    # Create appropriate datasets based on model type
    if is_encoder_decoder:
        train_dataset = CLIPEncoderDecoderDataset(sentence_file, train_ids_file, tokenizer, max_length)
        val_dataset = CLIPEncoderDecoderDataset(sentence_file, val_ids_file, tokenizer, max_length)
        test_dataset = CLIPEncoderDecoderDataset(sentence_file, test_ids_file, tokenizer, max_length)
    else:
        train_dataset = CLIPEncoderOnlyDataset(sentence_file, train_ids_file, tokenizer, max_length)
        val_dataset = CLIPEncoderOnlyDataset(sentence_file, val_ids_file, tokenizer, max_length)
        test_dataset = CLIPEncoderOnlyDataset(sentence_file, test_ids_file, tokenizer, max_length)

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

    logger.info(f"Prepared CLIP dataloaders with batch size {batch_size}")
    logger.info(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples, Test: {len(test_dataset)} examples")

    return train_loader, val_loader, test_loader


def analyze_clip_dataset(data_dir: str) -> Dict:
    """
    Analyze CLIP dataset and print statistics.

    Args:
        data_dir: Directory containing CLIP files

    Returns:
        Dictionary of dataset statistics
    """
    logger.info("Analyzing CLIP dataset")

    # Load data files
    sentence_file = os.path.join(data_dir, "clip", "full", "sentence_level.csv")
    train_ids_file = os.path.join(data_dir, "clip", "full", "train_ids.csv")
    val_ids_file = os.path.join(data_dir, "clip", "full", "val_ids.csv")
    test_ids_file = os.path.join(data_dir, "clip", "full", "test_ids.csv")

    # Load sentence data
    sentences_df = pd.read_csv(sentence_file)

    # Find ID column in sentences
    id_col = None
    for col in sentences_df.columns:
        if 'id' in col.lower():
            id_col = col
            break

    if id_col is None:
        logger.error("Could not find ID column in sentence file")
        return {}

    # Load ID files
    try:
        train_ids = pd.read_csv(train_ids_file)
    except:
        train_ids = pd.read_csv(train_ids_file, header=None, names=['doc_id'])

    try:
        val_ids = pd.read_csv(val_ids_file)
    except:
        val_ids = pd.read_csv(val_ids_file, header=None, names=['doc_id'])

    try:
        test_ids = pd.read_csv(test_ids_file)
    except:
        test_ids = pd.read_csv(test_ids_file, header=None, names=['doc_id'])

    # Filter sentences for each split
    train_notes = set(train_ids['doc_id'].values)
    val_notes = set(val_ids['doc_id'].values)
    test_notes = set(test_ids['doc_id'].values)

    train_sentences = sentences_df[sentences_df[id_col].isin(train_notes)]
    val_sentences = sentences_df[sentences_df[id_col].isin(val_notes)]
    test_sentences = sentences_df[sentences_df[id_col].isin(test_notes)]

    # Process labels if needed
    if 'labels' in sentences_df.columns:
        # Create label columns for statistics
        for label in CLIP_LABELS:
            sentences_df[label] = sentences_df['labels'].str.lower().str.contains(label.lower()).astype(int)
            train_sentences[label] = train_sentences['labels'].str.lower().str.contains(label.lower()).astype(int)
            val_sentences[label] = val_sentences['labels'].str.lower().str.contains(label.lower()).astype(int)
            test_sentences[label] = test_sentences['labels'].str.lower().str.contains(label.lower()).astype(int)

    # Calculate label statistics
    label_counts = {label: int(train_sentences[label].sum()) 
                  for label in CLIP_LABELS if label in train_sentences.columns}

    # Calculate sentence length statistics
    train_sentence_lengths = train_sentences['sentence'].str.split().str.len()

    # Calculate label co-occurrence
    label_cooccurrence = np.zeros((len(CLIP_LABELS), len(CLIP_LABELS)))
    for i, label1 in enumerate(CLIP_LABELS):
        if label1 not in train_sentences.columns:
            continue
        for j, label2 in enumerate(CLIP_LABELS):
            if label2 not in train_sentences.columns:
                continue
            # Count co-occurrences
            cooccur = (train_sentences[label1] & train_sentences[label2]).sum()
            label_cooccurrence[i, j] = cooccur

    # Create statistics dictionary
    stats = {
        'total_sentences': len(sentences_df),
        'train_sentences': len(train_sentences),
        'val_sentences': len(val_sentences),
        'test_sentences': len(test_sentences),
        'train_notes': len(train_notes),
        'val_notes': len(val_notes),
        'test_notes': len(test_notes),
        'label_counts': label_counts,
        'avg_sentence_length': train_sentence_lengths.mean(),
        'max_sentence_length': train_sentence_lengths.max(),
        'label_cooccurrence': label_cooccurrence.tolist()
    }

    # Log statistics
    logger.info(f"CLIP dataset statistics:")
    logger.info(f"Total sentences: {stats['total_sentences']}")
    logger.info(f"Train split: {stats['train_sentences']} sentences from {stats['train_notes']} notes")
    logger.info(f"Validation split: {stats['val_sentences']} sentences from {stats['val_notes']} notes")
    logger.info(f"Test split: {stats['test_sentences']} sentences from {stats['test_notes']} notes")
    logger.info(f"Label distribution: {stats['label_counts']}")
    logger.info(f"Average sentence length: {stats['avg_sentence_length']:.1f} words")

    return stats


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example usage
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    sentence_file = "data/clip/full/sentence_level.csv"
    train_ids_file = "data/clip/full/train_ids.csv"

    # Test dataset class
    dataset = CLIPEncoderOnlyDataset(sentence_file, train_ids_file, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample item: {dataset[0]}")
    print(f"Label distribution: {dataset.get_label_distribution()}")