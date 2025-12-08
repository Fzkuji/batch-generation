"""
Quick demo script to test cross-batch generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cross_batch_generator import CrossBatchGenerator


def demo():
    print("Cross-Batch Generation Demo")
    print("=" * 50)

    # Use a small model for quick testing
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create generator
    generator = CrossBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        mix_method="mixer",  # Use simpler mixer for demo
        device=device,
    )

    # Test prompts - related questions that could benefit from shared context
    prompts = [
        "The capital of France is",
        "Paris is famous for",
        "The Eiffel Tower is located in",
        "French cuisine includes",
    ]

    print("\nGenerating with cross-batch interaction...")
    print("-" * 50)

    # Generate with cross-batch
    texts_cross = generator.generate_text(
        prompts,
        max_new_tokens=20,
        enable_cross_batch=True,
    )

    print("\nWith Cross-Batch Interaction:")
    for prompt, text in zip(prompts, texts_cross):
        print(f"  Input: {prompt}")
        print(f"  Output: {text}")
        print()

    # Generate without cross-batch
    print("-" * 50)
    texts_standard = generator.generate_text(
        prompts,
        max_new_tokens=20,
        enable_cross_batch=False,
    )

    print("\nWithout Cross-Batch (Standard):")
    for prompt, text in zip(prompts, texts_standard):
        print(f"  Input: {prompt}")
        print(f"  Output: {text}")
        print()

    print("=" * 50)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
