"""
SQuAD dataset evaluation for cross-batch generation.
"""

import re
import string
from typing import List, Dict, Any, Tuple
from collections import Counter

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from .cross_batch_generator import CrossBatchGenerator


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute EM and F1 metrics.

    Args:
        predictions: List of predicted answers
        references: List of lists of reference answers (multiple valid answers per question)

    Returns:
        Dictionary with 'exact_match' and 'f1' scores
    """
    em_scores = []
    f1_scores = []

    for pred, refs in zip(predictions, references):
        # Take max score across all valid answers
        em = max(exact_match_score(pred, ref) for ref in refs)
        f1 = max(f1_score(pred, ref) for ref in refs)

        em_scores.append(em)
        f1_scores.append(f1)

    return {
        "exact_match": 100.0 * sum(em_scores) / len(em_scores),
        "f1": 100.0 * sum(f1_scores) / len(f1_scores),
    }


def format_squad_prompt(context: str, question: str) -> str:
    """Format a SQuAD example as a prompt."""
    prompt = f"""Context: {context}

Question: {question}

Answer:"""
    return prompt


def extract_answer(generated_text: str, prompt: str) -> str:
    """Extract the answer from generated text."""
    # Remove the prompt from generated text
    if generated_text.startswith(prompt):
        answer = generated_text[len(prompt):].strip()
    else:
        # Try to find "Answer:" marker
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()

    # Take first line/sentence as answer
    answer = answer.split('\n')[0].strip()
    answer = answer.split('.')[0].strip()

    return answer


class SquadEvaluator:
    """Evaluator for SQuAD dataset with cross-batch generation."""

    def __init__(
        self,
        generator: CrossBatchGenerator,
        tokenizer: PreTrainedTokenizer,
        split: str = "validation",
        max_samples: int = None,
    ):
        self.generator = generator
        self.tokenizer = tokenizer
        self.split = split
        self.max_samples = max_samples

        # Load dataset
        self.dataset = load_dataset("squad", split=split)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def prepare_batch(
        self,
        batch_indices: List[int],
    ) -> Tuple[List[str], List[List[str]]]:
        """Prepare a batch of examples."""
        prompts = []
        references = []

        for idx in batch_indices:
            example = self.dataset[idx]
            prompt = format_squad_prompt(example["context"], example["question"])
            prompts.append(prompt)
            references.append(example["answers"]["text"])

        return prompts, references

    def evaluate(
        self,
        batch_size: int = 4,
        max_new_tokens: int = 32,
        enable_cross_batch: bool = True,
        show_progress: bool = True,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate on SQuAD dataset.

        Args:
            batch_size: Number of examples per batch
            max_new_tokens: Maximum tokens to generate
            enable_cross_batch: Whether to enable cross-batch interaction
            show_progress: Whether to show progress bar
            **gen_kwargs: Additional generation kwargs

        Returns:
            Dictionary with metrics and predictions
        """
        all_predictions = []
        all_references = []
        all_prompts = []

        num_samples = len(self.dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size

        iterator = range(0, num_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating", total=num_batches)

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))

            prompts, references = self.prepare_batch(batch_indices)

            # Tokenize
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Generate
            outputs = self.generator.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                enable_cross_batch=enable_cross_batch,
                **gen_kwargs,
            )

            # Decode
            generated_texts = self.tokenizer.batch_decode(
                outputs["sequences"],
                skip_special_tokens=True,
            )

            # Extract answers
            for prompt, gen_text, refs in zip(prompts, generated_texts, references):
                answer = extract_answer(gen_text, prompt)
                all_predictions.append(answer)
                all_references.append(refs)
                all_prompts.append(prompt)

        # Compute metrics
        metrics = compute_metrics(all_predictions, all_references)

        return {
            "metrics": metrics,
            "predictions": all_predictions,
            "references": all_references,
            "prompts": all_prompts,
        }


def run_comparison_eval(
    generator: CrossBatchGenerator,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_samples: int = 100,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """
    Run comparison between cross-batch and standard generation.

    Args:
        generator: The CrossBatchGenerator instance
        tokenizer: Tokenizer
        batch_size: Batch size for evaluation
        max_samples: Maximum samples to evaluate
        max_new_tokens: Maximum new tokens to generate

    Returns:
        Comparison results
    """
    evaluator = SquadEvaluator(
        generator=generator,
        tokenizer=tokenizer,
        split="validation",
        max_samples=max_samples,
    )

    print("Evaluating with cross-batch interaction...")
    results_cross_batch = evaluator.evaluate(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        enable_cross_batch=True,
    )

    print("\nEvaluating without cross-batch interaction...")
    results_standard = evaluator.evaluate(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        enable_cross_batch=False,
    )

    print("\n" + "=" * 50)
    print("Results Comparison")
    print("=" * 50)
    print(f"\nWith Cross-Batch Interaction:")
    print(f"  Exact Match: {results_cross_batch['metrics']['exact_match']:.2f}")
    print(f"  F1 Score: {results_cross_batch['metrics']['f1']:.2f}")

    print(f"\nWithout Cross-Batch Interaction (Standard):")
    print(f"  Exact Match: {results_standard['metrics']['exact_match']:.2f}")
    print(f"  F1 Score: {results_standard['metrics']['f1']:.2f}")

    print(f"\nDifference (Cross-Batch - Standard):")
    em_diff = results_cross_batch['metrics']['exact_match'] - results_standard['metrics']['exact_match']
    f1_diff = results_cross_batch['metrics']['f1'] - results_standard['metrics']['f1']
    print(f"  Exact Match: {em_diff:+.2f}")
    print(f"  F1 Score: {f1_diff:+.2f}")

    return {
        "cross_batch": results_cross_batch,
        "standard": results_standard,
        "difference": {
            "exact_match": em_diff,
            "f1": f1_diff,
        },
    }
