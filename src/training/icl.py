#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
In-Context Learning Module

This module provides implementations for in-context learning on all three tasks:
MedNLI, RadQA, and CLIP.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import logging
import random
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForSequenceClassification, 
    PreTrainedTokenizer, 
    PreTrainedModel
)

logger = logging.getLogger(__name__)


class InContextLearner:
    """Base class for in-context learning."""

    def __init__(
        self, 
        model_path: str, 
        device: Optional[str] = None, 
        max_length: int = 512,
        verbose: bool = True,
        batch_size: int = 8
    ):
        """
        Initialize the in-context learner.

        Args:
            model_path: Path or name of the pre-trained model
            device: Device to use for inference
            max_length: Maximum sequence length
            verbose: Whether to print progress
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.max_length = max_length
        self.verbose = verbose
        self.batch_size = batch_size

        # Initialize device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
            self.is_seq2seq = True
        except:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.is_seq2seq = False

        if self.verbose:
            logger.info(f"Loaded model: {model_path} on {self.device}")
            logger.info(f"Model is {'seq2seq' if self.is_seq2seq else 'encoder-only'}")

    def cleanup(self):
        """Free up memory by releasing model and tokenizer."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        if self.verbose:
            logger.info("Cleaned up model and tokenizer")


class MedNLIInContextLearner(InContextLearner):
    """In-context learner for MedNLI task."""

    def generate_prompt(
        self, 
        premise: str, 
        hypothesis: str, 
        examples: List[Dict[str, str]] = None
    ) -> str:
        """
        Generate a prompt for MedNLI task.

        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            examples: List of examples to include in prompt (few-shot)

        Returns:
            Prompt string for model
        """
        if examples:
            # Few-shot prompt
            prompt = "Determine if the hypothesis follows from the premise. Answer with entailment, neutral, or contradiction.\n\n"

            # Include examples
            for ex in examples:
                prompt += f"Premise: {ex['premise']}\n"
                prompt += f"Hypothesis: {ex['hypothesis']}\n"
                prompt += f"Answer: {ex['label']}\n\n"

            # Include the current example
            prompt += f"Premise: {premise}\n"
            prompt += f"Hypothesis: {hypothesis}\n"
            prompt += "Answer:"
        else:
            # Zero-shot prompt
            prompt = "Determine if the hypothesis follows from the premise. Answer with entailment, neutral, or contradiction.\n\n"
            prompt += f"Premise: {premise}\n"
            prompt += f"Hypothesis: {hypothesis}\n"
            prompt += "Answer:"

        return prompt

    def parse_response(self, response: str) -> str:
        """
        Parse model response to extract the predicted label.

        Args:
            response: Model's raw output

        Returns:
            Extracted label
        """
        # Clean response
        response = response.strip().lower()

        # Check for label keywords
        if "entailment" in response:
            return "entailment"
        elif "neutral" in response:
            return "neutral"
        elif "contradiction" in response:
            return "contradiction"
        else:
            # Default to neutral if unclear
            return "neutral"

    def evaluate_mednli(
        self, 
        test_file: str, 
        examples_file: Optional[str] = None,
        num_examples: int = 3,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on MedNLI task with in-context learning.

        Args:
            test_file: Path to test file (JSONL)
            examples_file: Path to examples file (JSONL) for in-context examples
            num_examples: Number of examples to include in prompt
            num_samples: Maximum number of test samples to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Load test data
        test_data = []
        with open(test_file, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))

        # Sample if requested
        if num_samples and num_samples < len(test_data):
            random.seed(42)  # For reproducibility
            test_data = random.sample(test_data, num_samples)

        # Load examples for in-context learning
        examples = []
        if examples_file and num_examples > 0:
            with open(examples_file, 'r') as f:
                all_examples = [json.loads(line) for line in f]

            # Convert to standard format
            examples = [
                {
                    'premise': ex['sentence1'],
                    'hypothesis': ex['sentence2'],
                    'label': ex['gold_label']
                }
                for ex in all_examples
            ]

            # Sample examples
            random.seed(42)  # For reproducibility
            examples = random.sample(examples, min(num_examples, len(examples)))

        # Evaluate
        predictions = []
        targets = []

        for i, example in enumerate(test_data):
            if self.verbose and i % 10 == 0:
                logger.info(f"Evaluating example {i}/{len(test_data)}")

            premise = example['sentence1']
            hypothesis = example['sentence2']
            label = example['gold_label']

            # Generate prompt
            prompt = self.generate_prompt(premise, hypothesis, examples if num_examples > 0 else None)

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    do_sample=False,
                    num_return_sequences=1
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse to get label
            prediction = self.parse_response(response)

            predictions.append(prediction)
            targets.append(label)

        # Compute metrics
        accuracy = accuracy_score(targets, predictions)

        # Get label indices
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        target_indices = [label_map[t] for t in targets]
        pred_indices = [label_map[p] for p in predictions]

        f1 = f1_score(target_indices, pred_indices, average='macro')
        precision = precision_score(target_indices, pred_indices, average='macro')
        recall = recall_score(target_indices, pred_indices, average='macro')

        # Return metrics
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "predictions": predictions,
            "targets": targets,
            "num_examples": num_examples,
            "num_samples": len(test_data)
        }


class RadQAInContextLearner(InContextLearner):
    """In-context learner for RadQA task."""

    def generate_prompt(
        self, 
        question: str, 
        context: str, 
        examples: List[Dict[str, str]] = None
    ) -> str:
        """
        Generate a prompt for RadQA task.

        Args:
            question: Question text
            context: Context text
            examples: List of examples to include in prompt (few-shot)

        Returns:
            Prompt string for model
        """
        if examples:
            # Few-shot prompt
            prompt = "Answer the question based on the given context. If the question cannot be answered from the context, respond with 'unanswerable'.\n\n"

            # Include examples
            for ex in examples:
                prompt += f"Context: {ex['context']}\n"
                prompt += f"Question: {ex['question']}\n"
                prompt += f"Answer: {ex['answer']}\n\n"

            # Include the current example
            prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            prompt += "Answer:"
        else:
            # Zero-shot prompt
            prompt = "Answer the question based on the given context. If the question cannot be answered from the context, respond with 'unanswerable'.\n\n"
            prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            prompt += "Answer:"

        return prompt

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for evaluation.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Convert to lowercase and strip whitespace
        text = text.lower().strip()

        # Remove punctuation
        text = ''.join(c for c in text if c.isalnum() or c.isspace())

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def calculate_f1(self, prediction: str, reference: str) -> float:
        """
        Calculate word-level F1 score.

        Args:
            prediction: Predicted answer
            reference: Reference answer

        Returns:
            F1 score
        """
        pred_tokens = self._normalize_text(prediction).split()
        ref_tokens = self._normalize_text(reference).split()

        if not pred_tokens or not ref_tokens:
            return 1.0 if not pred_tokens and not ref_tokens else 0.0

        # Count common tokens
        common = sum(1 for token in pred_tokens if token in ref_tokens)

        # Calculate precision and recall
        precision = common / len(pred_tokens) if pred_tokens else 0.0
        recall = common / len(ref_tokens) if ref_tokens else 0.0

        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return f1

    def calculate_exact_match(self, prediction: str, reference: str) -> int:
        """
        Calculate exact match.

        Args:
            prediction: Predicted answer
            reference: Reference answer

        Returns:
            1 if exact match, 0 otherwise
        """
        return int(self._normalize_text(prediction) == self._normalize_text(reference))

    def evaluate_radqa(
        self, 
        test_file: str, 
        examples_file: Optional[str] = None,
        num_examples: int = 3,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on RadQA task with in-context learning.

        Args:
            test_file: Path to test file (JSON)
            examples_file: Path to examples file (JSON) for in-context examples
            num_examples: Number of examples to include in prompt
            num_samples: Maximum number of test samples to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Load test data
        with open(test_file, 'r') as f:
            test_data_raw = json.load(f)

        # Extract QA pairs
        test_data = []
        for article in test_data_raw['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    qa_id = qa['id']

                    # Handle answerable/unanswerable questions
                    is_impossible = qa.get('is_impossible', False)
                    if not is_impossible and len(qa['answers']) > 0:
                        # Take the first answer
                        answer = qa['answers'][0]['text']
                    else:
                        # For unanswerable questions
                        answer = "unanswerable"

                    test_data.append({
                        'id': qa_id,
                        'question': question,
                        'context': context,
                        'answer': answer,
                        'is_impossible': is_impossible
                    })

        # Sample if requested
        if num_samples and num_samples < len(test_data):
            random.seed(42)  # For reproducibility
            test_data = random.sample(test_data, num_samples)

        # Load examples for in-context learning
        examples = []
        if examples_file and num_examples > 0:
            with open(examples_file, 'r') as f:
                examples_data_raw = json.load(f)

            # Extract QA pairs
            all_examples = []
            for article in examples_data_raw['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa['question']

                        # Handle answerable/unanswerable questions
                        is_impossible = qa.get('is_impossible', False)
                        if not is_impossible and len(qa['answers']) > 0:
                            # Take the first answer
                            answer = qa['answers'][0]['text']
                        else:
                            # For unanswerable questions
                            answer = "unanswerable"

                        all_examples.append({
                            'question': question,
                            'context': context,
                            'answer': answer
                        })

            # Sample examples
            random.seed(42)  # For reproducibility
            examples = random.sample(all_examples, min(num_examples, len(all_examples)))

        # Evaluate in batches
        predictions = []
        references = []
        qa_ids = []

        # Create batches
        batches = [test_data[i:i+self.batch_size] for i in range(0, len(test_data), self.batch_size)]

        for batch_idx, batch in enumerate(batches):
            if self.verbose:
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")

            batch_prompts = []
            batch_answers = []
            batch_ids = []

            for example in batch:
                # Generate prompt
                prompt = self.generate_prompt(
                    example['question'], 
                    example['context'], 
                    examples if num_examples > 0 else None
                )

                batch_prompts.append(prompt)
                batch_answers.append(example['answer'])
                batch_ids.append(example['id'])

            # Tokenize all prompts
            batch_inputs = self.tokenizer(
                batch_prompts, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            ).to(self.device)

            # Generate responses
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    **batch_inputs,
                    max_length=64,  # Answer should be relatively short
                    do_sample=False,
                    num_return_sequences=1
                )

            # Decode responses
            batch_responses = self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

            # Add to overall results
            predictions.extend(batch_responses)
            references.extend(batch_answers)
            qa_ids.extend(batch_ids)

        # Compute metrics
        exact_matches = [
            self.calculate_exact_match(pred, ref) 
            for pred, ref in zip(predictions, references)
        ]

        f1_scores = [
            self.calculate_f1(pred, ref) 
            for pred, ref in zip(predictions, references)
        ]

        # Compute averages
        exact_match = sum(exact_matches) / len(exact_matches)
        f1 = sum(f1_scores) / len(f1_scores)

        # Return metrics
        return {
            "exact_match": exact_match,
            "f1": f1,
            "predictions": predictions,
            "ground_truths": references,
            "example_ids": qa_ids,
            "num_examples": num_examples,
            "num_samples": len(test_data)
        }


class CLIPInContextLearner(InContextLearner):
    """In-context learner for CLIP task (multi-label classification)."""

    def generate_prompt(
        self, 
        sentence: str, 
        examples: List[Dict[str, Union[str, List[str]]]] = None
    ) -> str:
        """
        Generate a prompt for CLIP task.

        Args:
            sentence: Sentence to classify
            examples: List of examples to include in prompt (few-shot)

        Returns:
            Prompt string for model
        """
        # Define available labels
        CLIP_LABELS = [
            'appointment-related',
            'medication-related',
            'lab-related',
            'patient-instructions',
            'procedure-related',
            'imaging-related',
            'other'
        ]

        if examples:
            # Few-shot prompt
            prompt = "Classify the following clinical sentences. Choose from these categories: "
            prompt += ", ".join(CLIP_LABELS)
            prompt += ". A sentence can have multiple labels. List all applicable labels separated by commas.\n\n"

            # Include examples
            for ex in examples:
                prompt += f"Sentence: {ex['sentence']}\n"
                prompt += f"Labels: {', '.join(ex['labels']) if ex['labels'] else 'none'}\n\n"

            # Include the current example
            prompt += f"Sentence: {sentence}\n"
            prompt += "Labels:"
        else:
            # Zero-shot prompt
            prompt = "Classify the following clinical sentence. Choose from these categories: "
            prompt += ", ".join(CLIP_LABELS)
            prompt += ". A sentence can have multiple labels. List all applicable labels separated by commas.\n\n"
            prompt += f"Sentence: {sentence}\n"
            prompt += "Labels:"

        return prompt

    def parse_response(self, response: str) -> List[str]:
        """
        Parse model response to extract predicted labels.

        Args:
            response: Model's raw output

        Returns:
            List of extracted labels
        """
        # Define available labels
        CLIP_LABELS = [
            'appointment-related',
            'medication-related',
            'lab-related',
            'patient-instructions',
            'procedure-related',
            'imaging-related',
            'other'
        ]

        # Clean response
        response = response.strip().lower()

        # Handle "none" or empty response
        if response == "none" or not response:
            return []

        # Split by comma or newline and strip whitespace
        parts = [p.strip() for p in response.replace('\n', ',').split(',')]

        # Filter to include only valid labels
        valid_labels = []
        for part in parts:
            for label in CLIP_LABELS:
                if label.lower() in part:
                    valid_labels.append(label)
                    break

        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in valid_labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)

        return unique_labels

    def labels_to_multihot(self, labels: List[str]) -> List[int]:
        """
        Convert list of labels to multi-hot encoding.

        Args:
            labels: List of label strings

        Returns:
            Multi-hot encoding (list of 0s and 1s)
        """
        # Define available labels
        CLIP_LABELS = [
            'appointment-related',
            'medication-related',
            'lab-related',
            'patient-instructions',
            'procedure-related',
            'imaging-related',
            'other'
        ]

        # Create multi-hot encoding
        multihot = [0] * len(CLIP_LABELS)
        for i, label in enumerate(CLIP_LABELS):
            if label in labels:
                multihot[i] = 1

        return multihot

    def evaluate_clip(
        self, 
        sentence_file: str,
        ids_file: str,
        examples_file: Optional[str] = None,
        examples_ids_file: Optional[str] = None,
        num_examples: int = 3,
        max_examples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on CLIP task with in-context learning.

        Args:
            sentence_file: Path to sentences CSV file
            ids_file: Path to test IDs CSV file
            examples_file: Path to sentences CSV file for in-context examples
            examples_ids_file: Path to training IDs CSV file for in-context examples
            num_examples: Number of examples to include in prompt
            max_examples: Maximum number of test examples to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Load sentences
        sentences_df = pd.read_csv(sentence_file)

        # Find ID column
        id_col = None
        for col in sentences_df.columns:
            if 'id' in col.lower():
                id_col = col
                break

        if id_col is None:
            logger.error("Could not find ID column in sentence file")
            return {}

        # Load test IDs
        try:
            test_ids_df = pd.read_csv(ids_file)
        except:
            test_ids_df = pd.read_csv(ids_file, header=None, names=['doc_id'])

        # Filter sentences for test set
        test_ids = set(test_ids_df['doc_id'].values)
        test_sentences = sentences_df[sentences_df[id_col].isin(test_ids)]

        # Sample if requested
        if max_examples and max_examples < len(test_sentences):
            test_sentences = test_sentences.sample(max_examples, random_state=42)

        # Define label columns
        CLIP_LABELS = [
            'appointment-related',
            'medication-related',
            'lab-related',
            'patient-instructions',
            'procedure-related',
            'imaging-related',
            'other'
        ]

        # Process labels if needed
        if 'labels' in test_sentences.columns:
            # Parse labels from string
            test_sentences['parsed_labels'] = test_sentences['labels'].apply(
                lambda x: [label for label in CLIP_LABELS if label.lower() in str(x).lower()]
            )
        else:
            # Assume one-hot format
            test_sentences['parsed_labels'] = test_sentences.apply(
                lambda row: [label for i, label in enumerate(CLIP_LABELS) if label in row and row[label] == 1],
                axis=1
            )

        # Load examples for in-context learning
        examples = []
        if examples_file and examples_ids_file and num_examples > 0:
            # Load example IDs
            try:
                examples_ids_df = pd.read_csv(examples_ids_file)
            except:
                examples_ids_df = pd.read_csv(examples_ids_file, header=None, names=['doc_id'])

            # Filter sentences for examples
            example_ids = set(examples_ids_df['doc_id'].values)
            example_sentences = sentences_df[sentences_df[id_col].isin(example_ids)]

            # Process labels if needed
            if 'labels' in example_sentences.columns:
                # Parse labels from string
                example_sentences['parsed_labels'] = example_sentences['labels'].apply(
                    lambda x: [label for label in CLIP_LABELS if label.lower() in str(x).lower()]
                )
            else:
                # Assume one-hot format
                example_sentences['parsed_labels'] = example_sentences.apply(
                    lambda row: [label for i, label in enumerate(CLIP_LABELS) if label in row and row[label] == 1],
                    axis=1
                )

            # Sample examples
            if len(example_sentences) > 0:
                examples_sample = example_sentences.sample(min(num_examples, len(example_sentences)), random_state=42)
                examples = [
                    {
                        'sentence': row['sentence'],
                        'labels': row['parsed_labels']
                    }
                    for _, row in examples_sample.iterrows()
                ]

        # Evaluate in batches
        predictions = []
        ground_truths = []

        # Create batches
        test_data = [
            {
                'sentence': row['sentence'],
                'labels': row['parsed_labels']
            }
            for _, row in test_sentences.iterrows()
        ]

        batches = [test_data[i:i+self.batch_size] for i in range(0, len(test_data), self.batch_size)]

        for batch_idx, batch in enumerate(batches):
            if self.verbose:
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")

            batch_prompts = []
            batch_labels = []

            for example in batch:
                # Generate prompt
                prompt = self.generate_prompt(
                    example['sentence'], 
                    examples if num_examples > 0 else None
                )

                batch_prompts.append(prompt)
                batch_labels.append(example['labels'])

            # Tokenize all prompts
            batch_inputs = self.tokenizer(
                batch_prompts, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            ).to(self.device)

            # Generate responses
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    **batch_inputs,
                    max_length=64,  # Labels should be relatively short
                    do_sample=False,
                    num_return_sequences=1
                )

            # Decode responses
            batch_responses = self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

            # Parse responses to get labels
            batch_predictions = [self.parse_response(response) for response in batch_responses]

            # Add to overall results
            predictions.extend(batch_predictions)
            ground_truths.extend(batch_labels)

        # Convert to multi-hot for metric computation
        multihot_predictions = [self.labels_to_multihot(pred) for pred in predictions]
        multihot_ground_truths = [self.labels_to_multihot(truth) for truth in ground_truths]

        # Compute metrics
        micro_f1 = f1_score(multihot_ground_truths, multihot_predictions, average='micro')
        macro_f1 = f1_score(multihot_ground_truths, multihot_predictions, average='macro')

        # Per-class metrics
        per_class_f1 = f1_score(multihot_ground_truths, multihot_predictions, average=None)
        per_class_metrics = {}
        for i, label in enumerate(CLIP_LABELS):
            per_class_metrics[f"{label}_f1"] = per_class_f1[i]

        # Return metrics
        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            **per_class_metrics,
            "predictions": predictions,
            "ground_truths": ground_truths,
            "num_examples": num_examples,
            "num_samples": len(test_data)
        }