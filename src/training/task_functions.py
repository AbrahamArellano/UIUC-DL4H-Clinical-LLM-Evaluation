#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task-Specific Training and Evaluation Functions

This module provides task-specific functions for training and evaluation
of models on MedNLI, RadQA, and CLIP tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)
from transformers import PreTrainedModel


def mednli_task_fn(
    batch: Optional[Dict[str, torch.Tensor]],
    model: PreTrainedModel,
    training: bool = True,
    outputs: Optional[Any] = None,
    get_predictions: bool = False,
    compute_metrics: bool = False,
    return_labels: bool = False,
    labels: Optional[List] = None,
    predictions: Optional[List] = None
) -> Union[Dict[str, torch.Tensor], List, Dict[str, float], Tuple]:
    """
    Task-specific function for MedNLI.

    This function handles preprocessing, postprocessing, and metrics computation
    for the MedNLI natural language inference task.

    Args:
        batch: Input batch (can be None if compute_metrics=True)
        model: Model being used
        training: Whether this is training mode
        outputs: Optional model outputs for prediction extraction
        get_predictions: Whether to extract predictions
        compute_metrics: Whether to compute metrics from labels and predictions
        return_labels: Whether to return labels along with processed batch
        labels: Optional ground truth labels for metric computation
        predictions: Optional model predictions for metric computation

    Returns:
        Processed batch, predictions, metrics or (batch, labels) depending on arguments
    """
    # Check if we're computing metrics from existing predictions
    if compute_metrics and labels is not None and predictions is not None:
        # Convert to numpy arrays if needed
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')

        # Return metrics dictionary
        return {
            'accuracy': accuracy,
            'f1': f1_macro,
            'precision': precision,
            'recall': recall
        }

    # Extract predictions from model outputs
    if get_predictions and outputs is not None:
        if hasattr(model, 'config') and hasattr(model.config, 'is_encoder_decoder') and model.config.is_encoder_decoder:
            # For encoder-decoder models (T5), get string predictions
            generated = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=8
            )
            # Convert token IDs to text
            tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
            if tokenizer:
                # Decode the generated tokens to text
                pred_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]

                # Map text to label IDs
                                label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
                                predictions = [label_map.get(text.strip().lower(), 1) for text in pred_texts]  # Default to neutral
                                return predictions
                            else:
                                # If no tokenizer, return raw IDs
                                return generated.cpu().numpy()
                        else:
                            # For encoder-only models (RoBERTa, BioClinRoBERTa), get class predictions
                            logits = outputs.logits
                            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                            return predictions

                    # Return batch with labels if requested
                    if return_labels and batch is not None:
                        if 'labels' in batch:
                            return batch, batch['labels']
                        elif 'label_id' in batch:
                            return batch, batch['label_id']
                        else:
                            # If no labels found, return empty tensor
                            return batch, torch.tensor([])

                    # Default: return the batch unchanged
                    return batch


                def radqa_task_fn(
                    batch: Optional[Dict[str, torch.Tensor]],
                    model: PreTrainedModel,
                    training: bool = True,
                    outputs: Optional[Any] = None,
                    get_predictions: bool = False,
                    compute_metrics: bool = False,
                    return_labels: bool = False,
                    labels: Optional[List] = None,
                    predictions: Optional[List] = None
                ) -> Union[Dict[str, torch.Tensor], List, Dict[str, float], Tuple]:
                    """
                    Task-specific function for RadQA.

                    This function handles preprocessing, postprocessing, and metrics computation
                    for the RadQA question-answering task.

                    Args:
                        batch: Input batch (can be None if compute_metrics=True)
                        model: Model being used
                        training: Whether this is training mode
                        outputs: Optional model outputs for prediction extraction
                        get_predictions: Whether to extract predictions
                        compute_metrics: Whether to compute metrics from labels and predictions
                        return_labels: Whether to return labels along with processed batch
                        labels: Optional ground truth labels for metric computation
                        predictions: Optional model predictions for metric computation

                    Returns:
                        Processed batch, predictions, metrics or (batch, labels) depending on arguments
                    """
                    # Check if we're computing metrics from existing predictions
                    if compute_metrics and labels is not None and predictions is not None:
                        # Calculate exact match
                        exact_match = calculate_exact_match(predictions, labels)

                        # Calculate F1 score
                        f1 = calculate_squad_f1(predictions, labels)

                        # Return metrics dictionary
                        return {
                            'exact_match': exact_match,
                            'f1': f1
                        }

                    # Extract predictions from model outputs
                    if get_predictions and outputs is not None:
                        if hasattr(model, 'config') and hasattr(model.config, 'is_encoder_decoder') and model.config.is_encoder_decoder:
                            # For encoder-decoder models (T5), get generated text
                            generated = model.generate(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                max_length=64
                            )
                            # Convert token IDs to text
                            tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
                            if tokenizer:
                                # Decode the generated tokens to text
                                predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
                                return predictions
                            else:
                                # If no tokenizer, return raw IDs
                                return generated.cpu().numpy()
                        else:
                            # For encoder-only models (extractive QA), get spans
                            start_logits = outputs.start_logits
                            end_logits = outputs.end_logits

                            # Get most likely span
                            start_idx = torch.argmax(start_logits, dim=-1).cpu().numpy()
                            end_idx = torch.argmax(end_logits, dim=-1).cpu().numpy()

                            # Ensure valid spans (end >= start)
                            for i in range(len(start_idx)):
                                if end_idx[i] < start_idx[i]:
                                    end_idx[i] = start_idx[i]

                            # Extract answer spans using input_ids and indices
                            answers = []
                            for i, (start, end) in enumerate(zip(start_idx, end_idx)):
                                # Skip invalid spans
                                if start == 0 or end == 0:
                                    answers.append("")
                                    continue

                                # Get tokens for this span
                                input_ids = batch['input_ids'][i]
                                span_tokens = input_ids[start:end+1]

                                # Decode tokens to text
                                tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
                                if tokenizer:
                                    answer = tokenizer.decode(span_tokens, skip_special_tokens=True)
                                    answers.append(answer)
                                else:
                                    # If no tokenizer, return raw token IDs
                                    answers.append(str(span_tokens.cpu().numpy()))

                            return answers

                    # Return batch with labels if requested
                    if return_labels and batch is not None:
                        labels_data = []
                        if 'labels' in batch:
                            # For encoder-decoder models, labels are token IDs
                            tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
                            if tokenizer:
                                for label_ids in batch['labels']:
                                    # Remove -100 tokens (padding)
                                    valid_ids = label_ids[label_ids >= 0]
                                    label_text = tokenizer.decode(valid_ids, skip_special_tokens=True)
                                    labels_data.append(label_text)
                            else:
                                labels_data = batch['labels']
                        elif 'start_positions' in batch and 'end_positions' in batch:
                            # For extractive QA, we need to extract spans using start/end positions
                            for i, (start, end) in enumerate(zip(batch['start_positions'], batch['end_positions'])):
                                # Skip invalid spans
                                if start == 0 or end == 0:
                                    labels_data.append("")
                                    continue

                                # Get tokens for this span
                                input_ids = batch['input_ids'][i]
                                span_tokens = input_ids[start:end+1]

                                # Decode tokens to text
                                tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
                                if tokenizer:
                                    answer = tokenizer.decode(span_tokens, skip_special_tokens=True)
                                    labels_data.append(answer)
                                else:
                                    # If no tokenizer, use raw positions
                                    labels_data.append((start.item(), end.item()))

                        return batch, torch.tensor(labels_data) if isinstance(labels_data, list) else labels_data

                    # Default: return the batch unchanged
                    return batch


                def clip_task_fn(
                    batch: Optional[Dict[str, torch.Tensor]],
                    model: PreTrainedModel,
                    training: bool = True,
                    outputs: Optional[Any] = None,
                    get_predictions: bool = False,
                    compute_metrics: bool = False,
                    return_labels: bool = False,
                    labels: Optional[List] = None,
                    predictions: Optional[List] = None
                ) -> Union[Dict[str, torch.Tensor], List, Dict[str, float], Tuple]:
                    """
                    Task-specific function for CLIP (Clinical Language Inference Prediction).

                    This function handles preprocessing, postprocessing, and metrics computation
                    for the CLIP multi-label classification task.

                    Args:
                        batch: Input batch (can be None if compute_metrics=True)
                        model: Model being used
                        training: Whether this is training mode
                        outputs: Optional model outputs for prediction extraction
                        get_predictions: Whether to extract predictions
                        compute_metrics: Whether to compute metrics from labels and predictions
                        return_labels: Whether to return labels along with processed batch
                        labels: Optional ground truth labels for metric computation
                        predictions: Optional model predictions for metric computation

                    Returns:
                        Processed batch, predictions, metrics or (batch, labels) depending on arguments
                    """
                    # Define CLIP labels
                    CLIP_LABELS = [
                        'appointment-related',
                        'medication-related',
                        'lab-related',
                        'patient-instructions',
                        'procedure-related',
                        'imaging-related',
                        'other'
                    ]

                    # Check if we're computing metrics from existing predictions
                    if compute_metrics and labels is not None and predictions is not None:
                        # Ensure numpy arrays
                        if isinstance(labels, torch.Tensor):
                            labels = labels.cpu().numpy()
                        if isinstance(predictions, torch.Tensor):
                            predictions = predictions.cpu().numpy()

                        # Multi-label metrics
                        micro_f1 = f1_score(labels, predictions, average='micro')
                        macro_f1 = f1_score(labels, predictions, average='macro')
                        precision_micro = precision_score(labels, predictions, average='micro')
                        recall_micro = recall_score(labels, predictions, average='micro')

                        # Per-class metrics
                        class_f1 = f1_score(labels, predictions, average=None)
                        class_precision = precision_score(labels, predictions, average=None)
                        class_recall = recall_score(labels, predictions, average=None)

                        # Create per-class metrics dictionary
                        class_metrics = {}
                        for i, label in enumerate(CLIP_LABELS):
                            class_metrics[f"{label}_f1"] = class_f1[i]
                            class_metrics[f"{label}_precision"] = class_precision[i]
                            class_metrics[f"{label}_recall"] = class_recall[i]

                        # Combine metrics
                        metrics = {
                            'micro_f1': micro_f1,
                            'macro_f1': macro_f1,
                            'precision_micro': precision_micro,
                            'recall_micro': recall_micro,
                            **class_metrics
                        }

                        return metrics

                    # Extract predictions from model outputs
                    if get_predictions and outputs is not None:
                        if hasattr(model, 'config') and hasattr(model.config, 'is_encoder_decoder') and model.config.is_encoder_decoder:
                            # For encoder-decoder models (T5), get generated text
                            generated = model.generate(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                max_length=128
                            )

                            # Convert token IDs to text
                            tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
                            if tokenizer:
                                # Decode the generated tokens to text
                                pred_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]

                                # Process predictions to extract labels
                                batch_predictions = []
                                for text in pred_texts:
                                    # Initialize multi-hot vector
                                    pred_vector = np.zeros(len(CLIP_LABELS), dtype=int)

                                    # Extract label mentions from text
                                    text = text.lower()
                                    for i, label in enumerate(CLIP_LABELS):
                                        if label.lower() in text:
                                            pred_vector[i] = 1

                                    batch_predictions.append(pred_vector)

                                return np.array(batch_predictions)
                            else:
                                # If no tokenizer, return raw IDs
                                return generated.cpu().numpy()
                        else:
                            # For encoder-only models, get multi-label predictions
                            logits = outputs.logits
                            predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                            return predictions

                    # Return batch with labels if requested
                    if return_labels and batch is not None:
                        if 'labels' in batch:
                            return batch, batch['labels']
                        else:
                            # If no labels found, return empty tensor
                            return batch, torch.tensor([])

                    # Default: return the batch unchanged
                    return batch


                # Utility functions for RadQA evaluation
                def calculate_exact_match(predictions, references):
                    """
                    Calculate exact match score for QA task.

                    Args:
                        predictions: List of predicted answers
                        references: List of reference answers

                    Returns:
                        Exact match score
                    """
                    if not predictions or not references:
                        return 0.0

                    # Normalize both prediction and reference
                    normalized_predictions = [normalize_text(pred) for pred in predictions]
                    normalized_references = [normalize_text(ref) for ref in references]

                    # Calculate exact matches
                    exact_matches = sum(pred == ref for pred, ref in zip(normalized_predictions, normalized_references))

                    return exact_matches / len(predictions)


                def calculate_squad_f1(predictions, references):
                    """
                    Calculate word-level F1 score for QA task.

                    Args:
                        predictions: List of predicted answers
                        references: List of reference answers

                    Returns:
                        F1 score
                    """
                    if not predictions or not references:
                        return 0.0

                    f1_scores = []

                    for pred, ref in zip(predictions, references):
                        # Tokenize prediction and reference
                        pred_tokens = normalize_text(pred).split()
                        ref_tokens = normalize_text(ref).split()

                        # Handle empty strings
                        if not pred_tokens or not ref_tokens:
                            if not pred_tokens and not ref_tokens:
                                f1_scores.append(1.0)  # Both empty is a match
                            else:
                                f1_scores.append(0.0)  # Only one empty is a mismatch
                            continue

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

                        f1_scores.append(f1)

                    return sum(f1_scores) / len(f1_scores)


                def normalize_text(text):
                    """
                    Normalize text for QA evaluation.

                    Args:
                        text: Text to normalize

                    Returns:
                        Normalized text
                    """
                    if not isinstance(text, str):
                        return ""

                    # Convert to lowercase
                    text = text.lower()

                    # Remove punctuation and extra whitespace
                    text = ''.join(c for c in text if c.isalnum() or c.isspace())
                    text = ' '.join(text.split())

                    return text