"""
Main entry point for cross-batch generation experiments.
"""

import argparse
import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cross_batch_generator import CrossBatchGenerator
from src.cross_batch_attention import CrossBatchAttention, CrossBatchEmbeddingMixer
from src.squad_eval import SquadEvaluator, run_comparison_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-batch generation experiments")

    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    # Cross-batch settings
    parser.add_argument(
        "--mix_method",
        type=str,
        default="attention",
        choices=["attention", "mixer"],
        help="Cross-batch mixing method",
    )
    parser.add_argument(
        "--mix_layer",
        type=int,
        default=-1,
        help="Which layer's hidden state to mix (-1 for last)",
    )

    # Evaluation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum new tokens to generate",
    )

    # Generation settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--run_comparison",
        action="store_true",
        help="Run comparison between cross-batch and standard generation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained cross-batch module checkpoint",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Cross-Batch Generation Experiment")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {args.device}")
    print(f"  Mix Method: {args.mix_method}")
    print(f"  Mix Layer: {args.mix_layer}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Samples: {args.max_samples}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Create cross-batch module
    hidden_size = model.config.hidden_size
    if args.mix_method == "attention":
        cross_batch_module = CrossBatchAttention(hidden_size=hidden_size)
    else:
        cross_batch_module = CrossBatchEmbeddingMixer(hidden_size=hidden_size)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        cross_batch_module.load_state_dict(checkpoint["cross_batch_module"])
        # Also load lm_head if available
        if "lm_head" in checkpoint and hasattr(model, 'lm_head'):
            model.lm_head.load_state_dict(checkpoint["lm_head"])
            print("Loaded cross_batch_module + lm_head")
        else:
            print("Loaded cross_batch_module only")

    # Create generator
    print("Creating cross-batch generator...")
    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        cross_batch_module=cross_batch_module,
        mix_layer=args.mix_layer,
        device=args.device,
    )

    if args.run_comparison:
        # Run comparison evaluation
        results = run_comparison_eval(
            generator=generator,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        # Single evaluation with cross-batch
        print("\nRunning evaluation with cross-batch interaction...")
        evaluator = SquadEvaluator(
            generator=generator,
            tokenizer=tokenizer,
            split="validation",
            max_samples=args.max_samples,
        )

        results = evaluator.evaluate(
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            enable_cross_batch=True,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )

        print("\n" + "=" * 50)
        print("Results")
        print("=" * 50)
        print(f"Exact Match: {results['metrics']['exact_match']:.2f}")
        print(f"F1 Score: {results['metrics']['f1']:.2f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")

    # Prepare results for saving (remove non-serializable items)
    save_results = {
        "config": vars(args),
        "metrics": results.get("metrics") or {
            "cross_batch": results["cross_batch"]["metrics"],
            "standard": results["standard"]["metrics"],
            "difference": results["difference"],
        },
    }

    with open(output_file, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
