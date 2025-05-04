#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training module for "Do We Still Need Clinical Language Models?"

This module provides training pipelines and utilities for both fine-tuning and
in-context learning experiments.
"""

from .trainer import (
    TrainingConfig,
    train_model,
    evaluate_model
)

from .task_functions import (
    mednli_task_fn,
    radqa_task_fn,
    clip_task_fn
)

from .icl import (
    InContextLearner,
    MedNLIInContextLearner,
    RadQAInContextLearner,
    CLIPInContextLearner
)