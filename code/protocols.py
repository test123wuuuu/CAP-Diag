"""Idea 5 Phase A0' — Define R0/R1/R2 rewrite prompts + NLI scorer wrapper.

R0 (existing): fact-skeleton short rephrase (~32 tokens, verbatim entity+date+value)
  prompt: from experiments/idea6_r4/scripts/generate_dual_family.py (reuse)

R1: medium paraphrase (~80-120 tokens, NLI-preserved, key facts preserved)
R2: longer rewrite (~150-250 tokens, NLI-preserved, surface form may differ substantially)

Acceptance rule per rewrite:
  - NLI bidirectional entailment OR neutral (NOT contradiction)
  - Required fact keys (entity, date, value, etc.) verbatim or accepted-variant present
  - length within target band (R1: [50, 150], R2: [120, 320])
"""
import sys, json, re, hashlib
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------- prompts ----------------
R0_PROMPT = (
    "You are rewriting a single short evidence passage for a synthetic dataset. "
    "Rewrite the given passage in 2-3 sentences (about 50 to 90 tokens). "
    "Keep ALL facts EXACTLY the same: every number, name, date, place must "
    "appear unchanged. Do not add new facts. Do not hedge. Output ONLY the "
    "rewritten passage, no preamble.\n\n"
    "Original passage:\n{target}\n\n"
    "Rewritten passage:"
)

R1_PROMPT = (
    "Paraphrase and slightly elaborate on the following passage for a synthetic dataset. "
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
    "Rewrite the following passage as a longer version for a synthetic dataset. "
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
REGIME_LENGTH_BAND = {"R0": (20, 100), "R1": (45, 180), "R2": (100, 320)}  # in TOKENS


# ---------------- NLI scorer ----------------
class NLIScorer:
    def __init__(self, model_dir, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, torch_dtype=torch.float32
        ).to(device).eval()
        self.device = device
        self.label2id = self.model.config.label2id
        self.id_entail = self.label2id.get("entailment", 1)
        self.id_contra = self.label2id.get("contradiction", 0)
        self.id_neutral = self.label2id.get("neutral", 2)

    @torch.no_grad()
    def score(self, premise: str, hypothesis: str):
        enc = self.tok(premise, hypothesis, return_tensors="pt",
                       truncation=True, max_length=256).to(self.device)
        out = self.model(**enc).logits.softmax(-1).cpu()[0].tolist()
        return {"entail": out[self.id_entail],
                "contradiction": out[self.id_contra],
                "neutral": out[self.id_neutral]}

    @torch.no_grad()
    def score_batch(self, pairs, batch_size=16):
        """pairs: list[(premise, hypothesis)]. Returns list of dict."""
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            premises = [p[0] for p in batch]
            hyps = [p[1] for p in batch]
            enc = self.tok(premises, hyps, return_tensors="pt",
                           truncation=True, max_length=256, padding=True).to(self.device)
            out = self.model(**enc).logits.softmax(-1).cpu().tolist()
            for v in out:
                results.append({
                    "entail": v[self.id_entail],
                    "contradiction": v[self.id_contra],
                    "neutral": v[self.id_neutral],
                })
        return results

    def is_bidirectional_entailment(self, text_a, text_b, entail_threshold=0.5,
                                    contra_threshold=0.5):
        """Bidirectional: both directions must NOT be 'contradiction' with high prob.
        Strict definition (entailment in both directions) is too strict for paraphrases
        that contain factual statements; we use 'no contradiction in either direction'."""
        s_ab = self.score(text_a, text_b)
        s_ba = self.score(text_b, text_a)
        return (s_ab["contradiction"] < contra_threshold and
                s_ba["contradiction"] < contra_threshold), s_ab, s_ba


# ---------------- skeleton check (reused from R4) ----------------
MONTHS = ["January","February","March","April","May","June","July",
         "August","September","October","November","December"]


def _date_variants(d_str):
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", d_str)
    if not m: return [d_str]
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12): return [d_str]
    mn = MONTHS[mo-1]
    return [d_str, f"{mn} {d}, {y}", f"{mn} {d:02d}, {y}",
            f"{d} {mn} {y}", f"{d:02d} {mn} {y}"]


def _time_variants(t_str):
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(AM|PM)", t_str.upper())
    if not m: return [t_str]
    h, mi, ampm = int(m.group(1)), m.group(2), m.group(3)
    return [t_str, f"{h}:{mi} {ampm}", f"{h}:{mi} {ampm.lower()}",
            f"{h:02d}:{mi} {ampm}", f"{h} {ampm}", f"{h:02d}:{mi} {ampm.lower()}"]


def check_skeleton(text, required, fact_skel):
    for k in required:
        v = fact_skel.get(k)
        if v is None: continue
        s = str(v)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            variants = _date_variants(s)
        elif re.fullmatch(r"\d{1,2}:\d{2}\s*(AM|PM|am|pm)", s):
            variants = _time_variants(s)
        else:
            variants = [s]
        if not any(v in text for v in variants):
            return False
    return True
