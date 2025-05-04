    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    """
    Ablation Study Script for "Do We Still Need Clinical Language Models?"

    This script runs an ablation study on the Longformer model with various input lengths
    to evaluate the impact of longer contexts on performance.
    """

    import os
    import sys
    import json
    import logging
    import argparse
    import time
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        LongformerForSequenceClassification,
        LongformerTokenizer,
        get_linear_schedule_with_warmup
    )

    # Add src to path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # Set default paths and parameters
    DEFAULT_OUTPUT_DIR = "results/ablation"
    DEFAULT_DATA_DIR = "data/mednli/full"
    SEED = 42


    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='Run ablation study for Longformer on MedNLI.'
        )

        # Model arguments
        parser.add_argument('--model', type=str, default="allenai/longformer-base-4096",
                            help='Longformer model to use')

        # Data arguments
        parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                            help='Directory containing MedNLI data')

        # Training arguments
        parser.add_argument('--input_lengths', nargs='+', type=int, default=[512, 1024, 2048],
                            help='Input sequence lengths to test')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='Batch size for training')
        parser.add_argument('--learning_rate', type=float, default=2e-5,
                            help='Learning rate')
        parser.add_argument('--epochs', type=int, default=3,
                            help='Number of epochs')
        parser.add_argument('--warmup_ratio', type=float, default=0.1,
                            help='Ratio of warmup steps')

        # Output arguments
        parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                            help='Output directory for results')
        parser.add_argument('--no_save_model', action='store_true',
                            help='Do not save trained models')

        # Hardware arguments
        parser.add_argument('--fp16', action='store_true',
                            help='Use mixed precision training')
        parser.add_argument('--device', type=str, default=None,
                            help='Device to use (cuda or cpu)')

        return parser.parse_args()


    class MedNLILongDataset(Dataset):
        """MedNLI dataset for Longformer."""

        def __init__(self, file_path, tokenizer, max_length):
            """
            Initialize MedNLI dataset for Longformer.

            Args:
                file_path: Path to JSONL file with MedNLI data
                tokenizer: Longformer tokenizer
                max_length: Maximum sequence length
            """
            self.examples = []
            with open(file_path, 'r') as f:
                for line in f:
                    ex = json.loads(line)
                    self.examples.append(ex)

            self.tokenizer = tokenizer
            self.max_length = max_length
            self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]
            premise = ex['sentence1']
            hypothesis = ex['sentence2']
            label = self.label_map[ex['gold_label']]

            # Add extra newlines to increase sequence length for ablation study
            if len(premise.split()) < self.max_length // 2:
                # Pad premise with newlines to approach target length
                padding_factor = max(1, (self.max_length // 2) // len(premise.split()))
                premise = premise + "\n\n" * padding_factor

            encoding = self.tokenizer(
                premise,
                hypothesis,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Create global attention mask (for Longformer)
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            global_attention_mask = torch.zeros_like(attention_mask)

            # Set global attention on CLS token
            global_attention_mask[0] = 1

            # Set global attention on all <s> and </s> tokens
            sep_token_indices = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            for idx in sep_token_indices:
                global_attention_mask[idx] = 1

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'global_attention_mask': global_attention_mask,
                'labels': torch.tensor(label, dtype=torch.long)
            }


    def train_one_epoch(model, optimizer, loader, device, scaler=None):
        """
        Train model for one epoch.

        Args:
            model: Longformer model
            optimizer: Optimizer
            loader: DataLoader
            device: Device to use
            scaler: Gradient scaler for mixed precision

        Returns:
            Average loss
        """
        model.train()
        total_loss = 0

        for batch in loader:
            optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        global_attention_mask=batch['global_attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss

                # Backward pass with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    global_attention_mask=batch['global_attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)


    def evaluate(model, loader, device):
        """
        Evaluate model.

        Args:
            model: Longformer model
            loader: DataLoader
            device: Device to use

        Returns:
            Accuracy
        """
        model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    global_attention_mask=batch['global_attention_mask'],
                    labels=batch['labels']
                )

                # Get predictions
                logits = outputs.logits
                preds.append(logits.argmax(dim=-1).cpu().numpy())
                labels.append(batch['labels'].cpu().numpy())

        # Concatenate predictions and labels
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        # Calculate accuracy
        acc = (preds == labels).mean()

        return acc


    def run_ablation_experiment(args):
        """
        Run ablation experiment with different input lengths.

        Args:
            args: Command line arguments

        Returns:
            Results dictionary
        """
        # Set random seed for reproducibility
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Set device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {device}")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Prepare data paths
        train_file = os.path.join(args.data_dir, "train.jsonl")
        dev_file = os.path.join(args.data_dir, "dev.jsonl")

        # Check if files exist
        if not os.path.exists(train_file) or not os.path.exists(dev_file):
            logger.error(f"Data files not found: {train_file} or {dev_file}")
            return None

        # Initialize results
        results = []

        # Run experiment for each input length
        for input_len in args.input_lengths:
            logger.info(f"\n🧪 Running Longformer for max_input_length={input_len}")

            # Create model output directory
            model_dir = os.path.join(args.output_dir, f"longformer_input_len_{input_len}")
            os.makedirs(model_dir, exist_ok=True)

            # Load tokenizer and model
            tokenizer = LongformerTokenizer.from_pretrained(args.model)
            model = LongformerForSequenceClassification.from_pretrained(
                args.model, num_labels=3).to(device)

            # Enable gradient checkpointing to save memory
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

            # Create datasets and dataloaders
            train_dataset = MedNLILongDataset(train_file, tokenizer, input_len)
            val_dataset = MedNLILongDataset(dev_file, tokenizer, input_len)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

            # Initialize optimizer and scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            total_steps = len(train_loader) * args.epochs
            warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

            # Initialize scaler for mixed precision training
            scaler = GradScaler() if args.fp16 and torch.cuda.is_available() else None

            # Training loop
            best_val_acc = 0
            train_losses = []
            val_accs = []

            for epoch in range(args.epochs):
                # Train
                epoch_start_time = time.time()
                train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
                train_losses.append(train_loss)

                # Evaluate
                val_acc = evaluate(model, val_loader, device)
                val_accs.append(val_acc)

                # Update scheduler
                scheduler.step()

                # Log progress
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                          f"train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, "
                          f"time={epoch_time:.1f}s")

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                    # Save model if requested
                    if not args.no_save_model:
                        model.save_pretrained(os.path.join(model_dir, "best_model"))
                        tokenizer.save_pretrained(os.path.join(model_dir, "best_model"))
                        logger.info(f"Saved best model to {os.path.join(model_dir, 'best_model')}")

            # Save results
            result = {
                "input_length": input_len,
                "val_accuracy": best_val_acc,
                "train_losses": train_losses,
                "val_accuracies": val_accs
            }

            results.append(result)

            # Save experiment results
            with open(os.path.join(model_dir, "results.json"), 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f"Completed experiment for input_length={input_len}, best_val_acc={best_val_acc:.4f}")

            # Clean up to free memory
            del model
            del tokenizer
            torch.cuda.empty_cache()

        return results


    def plot_results(results, output_dir):
        """
        Plot and save results.

        Args:
            results: List of result dictionaries
            output_dir: Output directory
        """
        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Plot validation accuracy vs input length
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_df, x="input_length", y="val_accuracy", marker="o")
        plt.title("Longformer MedNLI: Validation Accuracy vs Input Length")
        plt.xlabel("Max Input Length")
        plt.ylabel("Validation Accuracy")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(output_dir, "longformer_mednli_input_length_ablation.png"))
        plt.close()

        # Save CSV
        results_df.to_csv(os.path.join(output_dir, "longformer_mednli_input_length_ablation.csv"), index=False)

        logger.info(f"Saved results plot and CSV to {output_dir}")


    def main():
        """Main function."""
        # Parse arguments
        args = parse_args()

        logger.info("Starting Longformer ablation experiment")
        logger.info(f"Input lengths: {args.input_lengths}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Output directory: {args.output_dir}")

        # Run experiment
        results = run_ablation_experiment(args)

        if results:
            # Plot and save results
            plot_results(results, args.output_dir)

            # Print final summary
            print("\nAblation Study Results:")
            print("----------------------")
            df = pd.DataFrame(results)
            print(df[["input_length", "val_accuracy"]].to_string(index=False))

            logger.info("✅ Longformer ablation experiment completed!")
        else:
            logger.error("❌ Ablation experiment failed.")


    if __name__ == "__main__":
        main()