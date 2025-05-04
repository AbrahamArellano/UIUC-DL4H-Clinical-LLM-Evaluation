#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data module for "Do We Still Need Clinical Language Models?"

This module provides dataset implementations and utilities for MedNLI, RadQA, and CLIP tasks.
"""

from .mednli import (
    MedNLIDataset, MedNLIEncoderOnlyDataset, MedNLIEncoderDecoderDataset,
    load_mednli_data, create_mednli_subsets, prepare_mednli_dataloaders, analyze_mednli_dataset
)

from .radqa import (
    RadQADataset, RadQAEncoderOnlyDataset, RadQAEncoderDecoderDataset,
    load_radqa_data, create_radqa_subsets, prepare_radqa_dataloaders, analyze_radqa_dataset
)

try:
    from .clip import (
        CLIPDataset, CLIPEncoderOnlyDataset, CLIPEncoderDecoderDataset,
        load_clip_data, create_clip_subsets, prepare_clip_dataloaders, analyze_clip_dataset
    )
except ImportError:
    # CLIP module may not be available
    pass

from .utils import (
    verify_dataset_integrity, compute_file_hash, get_task_paths, create_directory_structure
)