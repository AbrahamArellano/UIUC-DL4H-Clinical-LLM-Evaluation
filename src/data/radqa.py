#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RadQA Dataset Module

This module provides dataset classes and utilities for the RadQA task.
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


class RadQADataset(Dataset):
    """Base class for RadQA dataset."""

    def __init__(self, json_file: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Initialize RadQA dataset.

        Args:
            json_file: Path to JSON file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.examples = []

        # Load data from JSON file (SQuAD format)
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Process examples
        for article in self.data['data']:
            article_id = article.get('title', '')
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    qa_id = qa['id']

                    # Handle answerable/unanswerable questions
                    is_impossible = qa.get('is_impossible', False)
                    if not is_impossible and len(qa['answers']) > 0:
                        # Take the first answer (could implement handling multiple answers)
                        answer = qa['answers'][0]
                        answer_text = answer['text']
                        answer_start = answer['answer_start']
                    else:
                        # For unanswerable questions
                        answer_text = ""
                        answer_start = -1

                    self.examples.append({
                        'id': qa_id,
                        'article_id': article_id,
                        'question': question,
                        'context': context,
                        'answer_text': answer_text,
                        'answer_start': answer_start,
                        'is_impossible': is_impossible
                    })

        logger.info(f"Loaded {len(self.examples)} examples from {json_file}")

    def __len__(self):
        return len(self.examples)

    def get_stats(self):
        """Return dataset statistics."""
        stats = {
            'total_examples': len(self.examples),
            'answerable': sum(1 for ex in self.examples if not ex['is_impossible']),
            'unanswerable': sum(1 for ex in self.examples if ex['is_impossible']),
            'avg_context_length': np.mean([len(ex['context'].split()) for ex in self.examples]),
            'avg_question_length': np.mean([len(ex['question'].split()) for ex in self.examples]),
            'avg_answer_length': np.mean([len(ex['answer_text'].split())
                                         for ex in self.examples if not ex['is_impossible'] and ex['answer_text']])
        }
        return stats


class RadQAEncoderOnlyDataset(RadQADataset):
    """RadQA dataset for encoder-only models (extractive QA)."""

    def __init__(self, json_file: str, tokenizer: PreTrainedTokenizer, max_length: int = 512, doc_stride: int = 128):
        """
        Initialize RadQA dataset for encoder-only models.

        Args:
            json_file: Path to JSON file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            doc_stride: Stride for splitting long documents
        """
        super().__init__(json_file, tokenizer, max_length)
        self.doc_stride = doc_stride

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        question = example['question']
        context = example['context']
        answer_text = example['answer_text']
        answer_start_char = example['answer_start']
        is_impossible = example['is_impossible']

        # Tokenize question and context
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation='only_second',  # Truncate only the context if needed
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        # Get offset mapping to map character positions to token positions
        offset_mappings = encoding.pop('offset_mapping')

        # Select the first window only (for simplicity)
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        offset_mapping = offset_mappings[0]

        # Initialize answer positions
        start_position = 0
        end_position = 0

        # Find the token positions for the answer if answerable
        if not is_impossible and answer_start_char >= 0:
            # Find the index where the context starts (after question + special tokens)
            sequence_ids = encoding.sequence_ids(0)
            context_start = sequence_ids.index(1) if 1 in sequence_ids else 0

            # Find the answer start and end in token space
            start_char = answer_start_char
            end_char = answer_start_char + len(answer_text)

            # Find start token position
            token_start_index = 0
            while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index -= 1

            # Find end token position
            token_end_index = token_start_index
            while token_end_index < len(offset_mapping) and offset_mapping[token_end_index][1] < end_char:
                token_end_index += 1

            # Adjust if answer is truncated
            if token_start_index >= self.max_length or token_end_index >= self.max_length:
                start_position = 0
                end_position = 0
            else:
                start_position = token_start_index
                end_position = token_end_index

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long),
            'example_id': example['id']
        }


class RadQAEncoderDecoderDataset(RadQADataset):
    """RadQA dataset for encoder-decoder models (generative QA)."""

    def __init__(self, json_file: str, tokenizer: PreTrainedTokenizer, max_length: int = 512, answer_max_length: int = 64):
        """
        Initialize RadQA dataset for encoder-decoder models.

        Args:
            json_file: Path to JSON file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length for input
            answer_max_length: Maximum length for answer
        """
        super().__init__(json_file, tokenizer, max_length)
        self.answer_max_length = answer_max_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        question = example['question']
        context = example['context']
        answer_text = example['answer_text']
        is_impossible = example['is_impossible']

        # Format for T5: "question: {question} context: {context}"
        input_text = f"question: {question} context: {context}"

        # For unanswerable questions
        if is_impossible or not answer_text:
            target_text = "unanswerable"
        else:
            target_text = answer_text

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
            max_length=self.answer_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract tensors
        input_ids = input_encoding['input_ids'].squeeze(0)
        attention_mask = input_encoding['attention_mask'].squeeze(0)
        labels = target_encoding['input_ids'].squeeze(0)

        # Replace padding token ids with -100 in labels (ignored in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'example_id': example['id']
        }


def load_radqa_data(data_dir: str, data_fraction: int = 100) -> Tuple[str, str, str]:
    """
    Load RadQA data files for the specified data fraction.

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
    radqa_dir = os.path.join(data_dir, "radqa", subset_dir)
    train_file = os.path.join(radqa_dir, "train.json")
    dev_file = os.path.join(radqa_dir, "dev.json")
    test_file = os.path.join(radqa_dir, "test.json")

    # Verify files exist
    for file_path in [train_file, dev_file, test_file]:
        if not os.path.exists(file_path):
            logger.warning(f"Data file not found: {file_path}")

    return train_file, dev_file, test_file


def create_radqa_subsets(source_dir: str, output_dir: str, percentages: List[int] = [1, 5, 10, 25, 100]):
    """
    Create random subsets of RadQA dataset.

    Args:
        source_dir: Directory containing original RadQA data
        output_dir: Directory to save subsets
        percentages: List of percentages to create
    """
    logger.info("Creating RadQA subsets")

    # Load training data
    train_file = os.path.join(source_dir, "train.json")
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    # Create directories for each subset
    for percentage in percentages:
        if percentage == 100:
            subset_name = "full"
        else:
            subset_name = f"{percentage}pct"

        subset_dir = os.path.join(output_dir, subset_name)
        os.makedirs(subset_dir, exist_ok=True)

        if subset_name == 'full':
            # Use full dataset
            subset_data = train_data
        else:
            # Create a random subset
            # Extract examples in flat structure for sampling
            examples = []
            for article in train_data['data']:
                article_id = article.get('title', '')
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        examples.append({
                            'article_id': article_id,
                            'qa_id': qa['id'],
                            'context': context,
                            'question': qa['question'],
                            'answers': qa.get('answers', []),
                            'is_impossible': qa.get('is_impossible', False)
                        })

            # Random sampling
            np.random.seed(42)
            sampled_indices = np.random.choice(
                len(examples), 
                size=int(len(examples) * percentage/100), 
                replace=False
            )
            sampled_examples = [examples[i] for i in sampled_indices]

            # Convert back to SQuAD format
            subset_data = {'data': []}
            article_groups = {}

            for example in sampled_examples:
                article_id = example['article_id']
                if article_id not in article_groups:
                    article_groups[article_id] = {'title': article_id, 'paragraphs': {}}

                context = example['context']
                if context not in article_groups[article_id]['paragraphs']:
                    article_groups[article_id]['paragraphs'][context] = {'context': context, 'qas': []}

                # Add QA pair
                qa = {
                    'id': example['qa_id'],
                    'question': example['question']
                }

                if example['is_impossible']:
                    qa['is_impossible'] = True
                    qa['answers'] = []
                else:
                    qa['answers'] = example['answers']

                article_groups[article_id]['paragraphs'][context]['qas'].append(qa)

            # Convert to final format
            for article_id, article in article_groups.items():
                paragraphs = []
                for context, para in article['paragraphs'].items():
                    paragraphs.append({
                        'context': context,
                        'qas': para['qas']
                    })

                subset_data['data'].append({
                    'title': article_id,
                    'paragraphs': paragraphs
                })

        # Save subset
        subset_train_file = os.path.join(subset_dir, "train.json")
        with open(subset_train_file, 'w') as f:
            json.dump(subset_data, f, indent=2)

        # Copy dev and test files
        for src_name, dst_name in [("dev.json", "dev.json"), ("test.json", "test.json")]:
            src_file = os.path.join(source_dir, src_name)
            dst_file = os.path.join(subset_dir, dst_name)
            with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                dst.write(src.read())

        # Count examples in subset
        qa_count = 0
        for article in subset_data['data']:
            for paragraph in article['paragraphs']:
                qa_count += len(paragraph['qas'])

        logger.info(f"Created {subset_name} subset with {qa_count} QA pairs")


def prepare_radqa_dataloaders(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    data_fraction: str = 'full',
    batch_size: Optional[int] = None,
    max_length: int = 512,
    data_dir: str = 'data'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare DataLoaders for RadQA task.

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
    train_file, dev_file, test_file = load_radqa_data(data_dir, 
                                                    int(data_fraction.rstrip('pct')) if data_fraction != 'full' else 100)

    # Determine if we're using an encoder-decoder model
    is_encoder_decoder = any(name in model_name.lower() for name in ['t5', 'bart'])

    # Set batch size based on model type if not specified
    if batch_size is None:
        if 'large' in model_name.lower():
            batch_size = 8  # Larger models need smaller batch size for QA
        elif 'base' in model_name.lower():
            batch_size = 16
        else:
            batch_size = 24

    # Create appropriate datasets based on model type
    if is_encoder_decoder:
        train_dataset = RadQAEncoderDecoderDataset(train_file, tokenizer, max_length)
        val_dataset = RadQAEncoderDecoderDataset(dev_file, tokenizer, max_length)
        test_dataset = RadQAEncoderDecoderDataset(test_file, tokenizer, max_length)
    else:
        train_dataset = RadQAEncoderOnlyDataset(train_file, tokenizer, max_length)
        val_dataset = RadQAEncoderOnlyDataset(dev_file, tokenizer, max_length)
        test_dataset = RadQAEncoderOnlyDataset(test_file, tokenizer, max_length)

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

    logger.info(f"Prepared RadQA dataloaders with batch size {batch_size}")
    logger.info(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples, Test: {len(test_dataset)} examples")

    return train_loader, val_loader, test_loader


def analyze_radqa_dataset(data_dir: str) -> Dict:
    """
    Analyze RadQA dataset and print statistics.

    Args:
        data_dir: Directory containing RadQA files

    Returns:
        Dictionary of dataset statistics
    """
    logger.info("Analyzing RadQA dataset")

    # Load data files
    train_file = os.path.join(data_dir, "radqa", "full", "train.json")
    dev_file = os.path.join(data_dir, "radqa", "full", "dev.json")
    test_file = os.path.join(data_dir, "radqa", "full", "test.json")

    # Load data
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(dev_file, 'r') as f:
        dev_data = json.load(f)

    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # Helper function to count examples and extract statistics
    def extract_stats(data):
        qa_count = 0
        answerable = 0
        unanswerable = 0
        question_lengths = []
        context_lengths = []
        answer_lengths = []

        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                context_lengths.append(len(context.split()))

                for qa in paragraph['qas']:
                    qa_count += 1
                    question_lengths.append(len(qa['question'].split()))

                    is_impossible = qa.get('is_impossible', False)
                    if is_impossible:
                        unanswerable += 1
                    else:
                        answerable += 1
                        if 'answers' in qa and qa['answers']:
                            answer_lengths.append(len(qa['answers'][0]['text'].split()))

        return {
            'qa_count': qa_count,
            'answerable': answerable,
            'unanswerable': unanswerable,
            'avg_question_length': np.mean(question_lengths),
            'avg_context_length': np.mean(context_lengths),
            'avg_answer_length': np.mean(answer_lengths) if answer_lengths else 0,
            'max_question_length': max(question_lengths) if question_lengths else 0,
            'max_context_length': max(context_lengths) if context_lengths else 0,
            'max_answer_length': max(answer_lengths) if answer_lengths else 0
        }

    # Extract statistics for each split
    train_stats = extract_stats(train_data)
    dev_stats = extract_stats(dev_data)
    test_stats = extract_stats(test_data)

    # Combine statistics
    stats = {
        'train': train_stats,
        'dev': dev_stats,
        'test': test_stats,
        'total_examples': train_stats['qa_count'] + dev_stats['qa_count'] + test_stats['qa_count']
    }

    # Log statistics
    logger.info(f"RadQA dataset statistics:")
    logger.info(f"Train set: {stats['train']['qa_count']} questions "
              f"({stats['train']['answerable']} answerable, {stats['train']['unanswerable']} unanswerable)")
    logger.info(f"Dev set: {stats['dev']['qa_count']} questions")
    logger.info(f"Test set: {stats['test']['qa_count']} questions")
    logger.info(f"Average question length: {stats['train']['avg_question_length']:.1f} words")
    logger.info(f"Average context length: {stats['train']['avg_context_length']:.1f} words")
    logger.info(f"Average answer length: {stats['train']['avg_answer_length']:.1f} words")

    return stats


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example usage
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_file = "data/radqa/full/train.json"

    # Test dataset class
    dataset = RadQAEncoderOnlyDataset(train_file, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample item: {dataset[0]}")
    print(f"Dataset stats: {dataset.get_stats()}")