"""
Training script for cross-batch module.
"""

import argparse
import torch

from src.trainer import train_cross_batch_module


def parse_args():
    parser = argparse.ArgumentParser(description="Train cross-batch module")

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--mix_method",
        type=str,
        default="attention",
        choices=["attention", "mixer"],
        help="Cross-batch mixing method",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum training samples",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Cross-Batch Module Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Mix Method: {args.mix_method}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Max Samples: {args.max_samples}")
    print(f"  Device: {args.device}")
    print()

    history = train_cross_batch_module(
        model_name=args.model_name,
        mix_method=args.mix_method,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Metrics:")
    print(f"  Final Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Baseline: {history['baseline_loss'][-1]:.4f}")
    print(f"  Final Improvement: {history['improvement'][-1]:.4f}")
    print(f"\nCheckpoints saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
