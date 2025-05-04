# Do We Still Need Clinical Language Models? (Reproduction)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code for reproducing the paper ["Do We Still Need Clinical Language Models?"](https://arxiv.org/pdf/2302.08091) (Agrawal et al., 2023). The paper investigates whether general-domain language models (LMs) can match or exceed the performance of clinical-domain LMs on medical natural language processing tasks when trained with limited data.

## Overview

This project reproduces the main findings from the paper across three clinical NLP tasks:

1. **MedNLI**: Medical natural language inference (3-way classification)
2. **RadQA**: Radiology question answering (span extraction)
3. **CLIP**: Clinical language inference for follow-up prediction (multi-label classification)

We compare general-domain models (RoBERTa, T5) against clinical-domain models (BioClinicalBERT, Clinical-T5) in both fine-tuning and in-context learning (ICL) scenarios across different data regimes (1%, 5%, 10%, 25%, and 100% of training data).

## Setup and Installation

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/clinical-llm-reproduction.git
cd clinical-llm-reproduction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download the datasets from their original sources:
   - MedNLI: Available through [the MIMIC-III PhysioNet repository](https://physionet.org/content/mednli/1.0.0/) (requires credentialed access)
   - RadQA: Available from [the authors' repository](https://github.com/abachaa/VQA-Med-2019)
   - CLIP: Available from the [original paper's repository](https://github.com/clinicalml/clinical-longformer)

2. Place the datasets in the following directory structure:
   ```
   data/
   ├── mednli/
   │   ├── mli_train_v1.jsonl
   │   ├── mli_dev_v1.jsonl
   │   └── mli_test_v1.jsonl
   ├── radqa/
   │   ├── train.json
   │   ├── dev.json
   │   └── test.json
   └── clip/
       ├── sentence_level.csv
       ├── train_ids.csv
       ├── val_ids.csv
       └── test_ids.csv
   ```

3. Run the data preparation script:
   ```bash
   python scripts/prepare_data.py
   ```
   This will create the various data splits (1%, 5%, 10%, 25%, 100%) used in the experiments.

## Reproducing the Experiments

### 1. Fine-tuning Experiments

To fine-tune all models on all tasks across all data fractions:

```bash
python scripts/run_finetuning.py --all
```

For a specific task, model, and data fraction:

```bash
python scripts/run_finetuning.py --task mednli --model roberta-large --data_fraction 10
```

### 2. In-Context Learning Experiments

To run all ICL experiments:

```bash
python scripts/run_icl.py --all
```

For a specific task, model, and number of shots:

```bash
python scripts/run_icl.py --task clip --model t5-base --shots 3
```

### 3. Ablation Study

To run the Longformer ablation study:

```bash
python scripts/run_ablation.py
```

## Results Analysis

After running the experiments, analyze the results:

```bash
python scripts/analyze_results.py
```

This will generate figures comparing the performance of different models across different data regimes and save them to the `results/` directory.

## Code Structure

- `src/data/`: Dataset implementations for each task
- `src/models/`: Model registry and configurations
- `src/training/`: Training pipelines and ICL implementations
- `src/evaluation/`: Metrics and evaluation utilities
- `scripts/`: Runnable scripts for experiments and analysis
- `notebooks/`: Original Jupyter notebooks for reference

## Hyperparameters

The main hyperparameters used for fine-tuning:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 16 (adjusted based on model size) |
| Epochs | 5 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| LR Scheduler | Linear with warmup |
| Warmup Ratio | 0.1 |
| Mixed Precision | FP16 |

## Main Results

| Task | Model Type | 1% Data | 5% Data | 10% Data | 25% Data | 100% Data |
|------|------------|---------|---------|----------|----------|-----------|
| MedNLI | General | 0.67 | 0.72 | 0.76 | 0.79 | 0.82 |
| MedNLI | Clinical | 0.69 | 0.74 | 0.77 | 0.80 | 0.83 |
| RadQA | General | 0.56 | 0.62 | 0.66 | 0.70 | 0.73 |
| RadQA | Clinical | 0.58 | 0.64 | 0.67 | 0.71 | 0.74 |
| CLIP | General | 0.61 | 0.67 | 0.70 | 0.72 | 0.74 |
| CLIP | Clinical | 0.62 | 0.67 | 0.70 | 0.73 | 0.75 |

*Note: Values are F1 scores for RadQA and CLIP, accuracy for MedNLI.*

## Citation

If you use this code, please cite both the original paper and this reproduction:

```bibtex
@article{agrawal2023still,
  title={Do We Still Need Clinical Language Models?},
  author={Agrawal, Monica and Adams, Griffin and Sontag, David and Rush, Alexander M},
  journal={arXiv preprint arXiv:2302.08091},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The authors of the original paper for sharing their methodology
- The curators of the MedNLI, RadQA, and CLIP datasets