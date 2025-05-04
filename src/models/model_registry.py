#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Registry Module

This module provides a centralized model registry, unified loading logic, and configuration.
"""

import os
import json
import torch
import logging
from typing import Dict, Tuple, Any, Optional, List, Union

from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    AutoModelForSequenceClassification, AutoTokenizer,
    PreTrainedModel, PreTrainedTokenizer
)

logger = logging.getLogger(__name__)

# Model Registry with specifications
MODEL_REGISTRY = {
    "t5-base": {
        "version": "v1.0",
        "model_class": T5ForConditionalGeneration,
        "tokenizer_class": T5Tokenizer,
        "pretrained": "t5-base",
        "type": "general",
        "size": "base",
        "params": 220_000_000,
        "task_head": "seq2seq",
        "learning_rate": 1e-4,
        "batch_size": 24,
        "dropout": 0.1,
        "is_encoder_decoder": True,
        "notes": "Standard T5-Base for general tasks"
    },
    "t5-large": {
        "version": "v1.0",
        "model_class": T5ForConditionalGeneration,
        "tokenizer_class": T5Tokenizer,
        "pretrained": "t5-large",
        "type": "general",
        "size": "large",
        "params": 770_000_000,
        "task_head": "seq2seq",
        "learning_rate": 5e-5,
        "batch_size": 8,
        "dropout": 0.1,
        "is_encoder_decoder": True,
        "notes": "Standard T5-Large for general tasks"
    },
    "roberta-large": {
        "version": "v1.0",
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": RobertaTokenizer,
        "pretrained": "roberta-large",
        "type": "general",
        "size": "large",
        "params": 355_000_000,
        "task_head": "classification",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "dropout": 0.1,
        "is_encoder_decoder": False,
        "notes": "General-purpose RoBERTa-Large model for classification tasks"
    },
    "BioClinRoBERTa": {
        "version": "v1.0",
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "pretrained": "emilyalsentzer/Bio_ClinicalBERT",
        "type": "clinical",
        "size": "base",
        "params": 110_000_000,
        "task_head": "classification",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "dropout": 0.1,
        "is_encoder_decoder": False,
        "notes": "Clinical model built on Bio_ClinicalBERT"
    },
    "clinical-t5-base": {
        "version": "v1.0",
        "model_class": T5ForConditionalGeneration,
        "tokenizer_class": T5Tokenizer,
        "pretrained": "StanfordAIMI/clinical-t5-base",  # Update if source changes
        "type": "clinical",
        "size": "base",
        "params": 220_000_000,
        "task_head": "seq2seq",
        "learning_rate": 1e-4,
        "batch_size": 24,
        "dropout": 0.1,
        "is_encoder_decoder": True,
        "notes": "Clinical T5-Base model"
    },
    "google/flan-t5-base": {
        "version": "v1.0",
        "model_class": T5ForConditionalGeneration,
        "tokenizer_class": T5Tokenizer,
        "pretrained": "google/flan-t5-base",
        "type": "general",
        "size": "base",
        "params": 250_000_000,
        "task_head": "seq2seq",
        "learning_rate": 1e-4,
        "batch_size": 24,
        "dropout": 0.1,
        "is_encoder_decoder": True,
        "notes": "Instruction-tuned T5 model for ICL"
    },
    "google/flan-t5-large": {
        "version": "v1.0",
        "model_class": T5ForConditionalGeneration,
        "tokenizer_class": T5Tokenizer,
        "pretrained": "google/flan-t5-large",
        "type": "general",
        "size": "large",
        "params": 770_000_000,
        "task_head": "seq2seq",
        "learning_rate": 5e-5,
        "batch_size": 8,
        "dropout": 0.1,
        "is_encoder_decoder": True,
        "notes": "Instruction-tuned T5-Large model for ICL"
    }
}


def load_model(
    model_key: str, 
    num_labels: Optional[int] = None, 
    task: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
    """
    Load a model and tokenizer from the registry.

    Args:
        model_key: Name of the model in the registry
        num_labels: Number of labels for classification tasks
        task: Task type (for task-specific model configuration)

    Returns:
        model, tokenizer, config: Loaded model, tokenizer and configuration dictionary
    """
    if model_key not in MODEL_REGISTRY:
        # Try to find it in registry by checking if it's a model path
        for key, config in MODEL_REGISTRY.items():
            if config["pretrained"] == model_key:
                model_key = key
                break
        else:
            raise ValueError(f"Model {model_key} not found in registry")

    config = MODEL_REGISTRY[model_key]
    logger.info(f"Loading {model_key} model (type: {config['type']}, size: {config['size']})")

    # Set task-specific parameters if needed
    model_kwargs = {}
    if task == "mednli" and config['task_head'] == "classification":
        model_kwargs["num_labels"] = 3
    elif task == "clip" and config['task_head'] == "classification":
        model_kwargs["num_labels"] = 7
    elif num_labels is not None and not config['is_encoder_decoder']:
        model_kwargs["num_labels"] = num_labels

    # Load model and tokenizer
    try:
        # Handle potential memory optimization for large models
        if config["size"] == "large" and torch.cuda.is_available():
            if config['is_encoder_decoder']:
                # Memory-efficient loading for encoder-decoder models
                tokenizer = config["tokenizer_class"].from_pretrained(config["pretrained"])
                model = config["model_class"].from_pretrained(
                    config["pretrained"],
                    device_map="auto",
                    torch_dtype=torch.float16,
                    **model_kwargs
                )
            else:
                # Memory-efficient loading for encoder-only models
                tokenizer = config["tokenizer_class"].from_pretrained(config["pretrained"])
                model = config["model_class"].from_pretrained(
                    config["pretrained"],
                    device_map="auto",
                    torch_dtype=torch.float16,
                    **model_kwargs
                )
        else:
            # Standard loading for smaller models
            tokenizer = config["tokenizer_class"].from_pretrained(config["pretrained"])
            model = config["model_class"].from_pretrained(config["pretrained"], **model_kwargs)

        logger.info(f"Successfully loaded {model_key} with ~{config['params']:,} parameters")
        return model, tokenizer, config

    except Exception as e:
        logger.error(f"Failed to load {model_key}: {e}")
        raise


def get_model_info(model_key: str) -> Dict[str, Any]:
    """
    Get model information from the registry.

    Args:
        model_key: Name of the model in the registry

    Returns:
        Model configuration dictionary
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_key} not found in registry")

    return MODEL_REGISTRY[model_key]


def get_model_args_for_task(model_key: str, task: str) -> Dict[str, Any]:
    """
    Get task-specific model arguments.

    Args:
        model_key: Name of the model in the registry
        task: Task name

    Returns:
        Dictionary of task-specific model arguments
    """
    config = get_model_info(model_key)

    # Set number of labels based on task
    task_args = {}
    if config['task_head'] == "classification":
        if task == "mednli":
            task_args["num_labels"] = 3
        elif task == "clip":
            task_args["num_labels"] = 7

    return task_args


def save_finetuned_model(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    output_dir: str, 
    model_name: str, 
    task: str,
    data_fraction: Union[int, str]
) -> str:
    """
    Save a fine-tuned model and tokenizer.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        output_dir: Base output directory
        model_name: Name of the model
        task: Task name
        data_fraction: Data fraction used for training

    Returns:
        Path to saved model
    """
    # Create save directory
    model_dir = os.path.join(output_dir, task, model_name.replace('/', '_'), 
                           f"{data_fraction}pct" if isinstance(data_fraction, int) else data_fraction)
    os.makedirs(model_dir, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Save model configuration
    if model_name in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model_name].copy()
        config["fine_tuned"] = True
        config["task"] = task
        config["data_fraction"] = data_fraction

        with open(os.path.join(model_dir, "model_config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    logger.info(f"Saved fine-tuned model to {model_dir}")
    return model_dir


def load_finetuned_model(
    model_dir: str,
    task: str,
    device: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a fine-tuned model and tokenizer.

    Args:
        model_dir: Directory containing the saved model
        task: Task name
        device: Device to load the model to

    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    # Check if model_config.json exists
    config_file = os.path.join(model_dir, "model_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

        model_class = None
        tokenizer_class = None

        # Determine model and tokenizer classes
        if config.get("is_encoder_decoder", False):
            model_class = T5ForConditionalGeneration
            tokenizer_class = T5Tokenizer
        elif "roberta" in model_dir.lower() or "bert" in model_dir.lower():
            model_class = AutoModelForSequenceClassification
            tokenizer_class = AutoTokenizer
        else:
            # Default to Auto classes
            model_class = AutoModelForSequenceClassification
            tokenizer_class = AutoTokenizer
    else:
        # Default to Auto classes
        model_class = AutoModelForSequenceClassification
        tokenizer_class = AutoTokenizer

    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_dir)

    # Load model
    device_map = "auto" if device is None else device
    model = model_class.from_pretrained(
        model_dir, 
        device_map=device_map if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None
    )

    logger.info(f"Loaded fine-tuned model from {model_dir}")
    return model, tokenizer


def count_parameters(model: PreTrainedModel) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model: Model to analyze

    Returns:
        total_params, trainable_params: Total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable parameters: {trainable:,}")

    return total, trainable


def get_all_models(model_type: Optional[str] = None) -> List[str]:
    """
    Get all available models, optionally filtered by type.

    Args:
        model_type: Filter by model type (clinical, general)

    Returns:
        List of model names
    """
    if model_type:
        return [model for model, config in MODEL_REGISTRY.items() 
               if config.get("type") == model_type]
    else:
        return list(MODEL_REGISTRY.keys())


def log_model_config(model_key: str):
    """
    Log model configuration.

    Args:
        model_key: Name of the model in the registry
    """
    config = get_model_info(model_key)
    logger.info(f"\nModel Configuration for {model_key}:")
    for k, v in config.items():
        logger.info(f" - {k}: {v}")


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example usage
    for model_name in MODEL_REGISTRY:
        print(f"\nModel: {model_name}")
        info = get_model_info(model_name)
        print(f"  Type: {info['type']}")
        print(f"  Parameters: {info['params']:,}")
        print(f"  Batch size: {info['batch_size']}")