"""Core protocol definitions for rewrite regimes and NLI validation.

This module defines the three rewrite regimes (R0, R1, R2) used to generate
controlled text variations for watermark analysis, along with NLI-based
semantic preservation checks.
"""
import sys, json, re, hashlib
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------- Rewrite regime prompts ----------------
R0_PROMPT = (
    "You are rewriting a short evidence passage. "
    "Rewrite the given passage in 2-3 sentences (about 50 to 90 tokens). "
    "Keep ALL facts EXACTLY the same: every number, name, date, place must "
    "appear unchanged. Do not add new facts. Do not hedge. Output ONLY the "
    "rewritten passage, no preamble.\n\n"
    "Original passage:\n{target}\n\n"
    "Rewritten passage:"
)

R1_PROMPT = (
    "Paraphrase and slightly elaborate on the following passage. "
    "Write a multi-sentence rewrite (aim for 60 to 120 tokens). "
    "Loosen surface form, vary word choice and sentence structure, AND add a sentence "
    "of contextual explanation or restatement. BUT preserve every key fact: "
    "entities, numbers, dates, places, and quantities must appear unchanged. "
    "Do not invent new facts or different values. Do not hedge. "
    "Output ONLY the rewritten passage, no preamble.\n\n"
    "Original passage:\n{target}\n\n"
    "Rewritten passage:"
)

R2_PROMPT = (
    "Rewrite the following passage as a longer version. "
    "Write 6-10 sentences (about 150 to 250 tokens). "
    "You may add context, elaboration, and longer phrasing. "
    "PRESERVE every key fact unchanged: entities, numbers, dates, places, "
    "quantities. Surface form may differ substantially. Do not invent new facts "
    "or different values. Output ONLY the rewritten passage, no preamble.\n\n"
    "Original passage:\n{target}\n\n"
    "Rewritten passage:"
)

REGIME_PROMPTS = {"R0": R0_PROMPT, "R1": R1_PROMPT, "R2": R2_PROMPT}
REGIME_MAX_TOKENS = {"R0": 160, "R1": 240, "R2": 400}
REGIME_LENGTH_BAND = {"R0": (20, 100), "R1": (45, 180), "R2": (100, 320)}


# ---------------- NLI scorer for semantic preservation ----------------
class NLIScorer:
    """Bidirectional NLI scorer using DeBERTa-v3-large for contradiction detection."""

    def __init__(self, model_name="microsoft/deberta-v3-large", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    def score(self, premise, hypothesis):
        """Return contradiction score [0,1]. Higher means more contradictory."""
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # DeBERTa NLI: [contradiction, neutral, entailment]
        return probs[0].item()

    def bidirectional_contradiction(self, text_a, text_b, threshold=0.5):
        """Check if texts contradict in either direction."""
        score_ab = self.score(text_a, text_b)
        score_ba = self.score(text_b, text_a)
        return max(score_ab, score_ba) > threshold


# ---------------- Utility functions ----------------
def count_tokens(text, tokenizer=None):
    """Count tokens in text. Uses simple whitespace splitting if no tokenizer provided."""
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        return len(text.split())


def deterministic_seed(item_id, variant, base_seed=42):
    """Generate deterministic seed from item ID and variant for reproducibility."""
    h = hashlib.md5(f"{item_id}|{variant}|{base_seed}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)
