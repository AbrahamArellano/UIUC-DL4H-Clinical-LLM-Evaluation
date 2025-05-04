#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Models module for "Do We Still Need Clinical Language Models?"

This module provides model registry and utilities for loading and managing models.
"""

from .model_registry import (
    MODEL_REGISTRY, 
    load_model, 
    get_model_info, 
    get_model_args_for_task, 
    save_finetuned_model,
    load_finetuned_model,
    count_parameters,
    get_all_models,
    log_model_config
)