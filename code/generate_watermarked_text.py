"""Generate watermarked text across rewrite regimes for capacity analysis.

This script generates text using KGW, Unigram, or EXP watermarking schemes
across R0/R1/R2 regimes for measuring detector-statistic-accessible MI.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path, device="cuda"):
    """Load language model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def generate_with_watermark(model, tokenizer, prompt, watermark_config, max_tokens=200):
    """Generate text with specified watermark configuration.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        watermark_config: Dict with 'method' (kgw/uw/exp) and 'strength'
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text string
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Configure watermark logits processor based on method
    method = watermark_config.get("method", "kgw")
    strength = watermark_config.get("strength", 2.0)

    if method == "kgw":
        from watermark_processors import KGWLogitsProcessor
        processor = KGWLogitsProcessor(
            vocab_size=len(tokenizer),
            gamma=0.5,
            delta=strength,
            seeding_scheme="simple"
        )
    elif method == "uw":
        from watermark_processors import UniformLogitsProcessor
        processor = UniformLogitsProcessor(
            vocab_size=len(tokenizer),
            delta=strength
        )
    elif method == "exp":
        from watermark_processors import ExpLogitsProcessor
        processor = ExpLogitsProcessor(
            vocab_size=len(tokenizer),
            strength=strength
        )
    else:
        processor = None

    logits_processor = [processor] if processor else []

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=1.0,
        logits_processor=logits_processor
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from output
    generated_text = generated_text[len(prompt):].strip()

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate watermarked text")
    parser.add_argument("--model_path", default="./models/Qwen2.5-7B-Instruct",
                        help="Path to language model")
    parser.add_argument("--input", required=True,
                        help="Input JSONL with source passages")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--regime", choices=["R0", "R1", "R2"], required=True,
                        help="Rewrite regime")
    parser.add_argument("--watermark", choices=["kgw", "uw", "exp", "none"],
                        default="kgw", help="Watermark method")
    parser.add_argument("--strength", type=float, default=2.0,
                        help="Watermark strength parameter")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    # Load regime prompts
    from protocols_clean import REGIME_PROMPTS, REGIME_MAX_TOKENS

    prompt_template = REGIME_PROMPTS[args.regime]
    max_tokens = REGIME_MAX_TOKENS[args.regime]

    # Load input passages
    with open(args.input) as f:
        passages = [json.loads(line) for line in f]

    results = []

    watermark_config = {
        "method": args.watermark if args.watermark != "none" else None,
        "strength": args.strength
    }

    for i, item in enumerate(passages):
        source_text = item.get("text", item.get("passage", ""))

        # Format prompt for this regime
        prompt = prompt_template.format(target=source_text)

        # Generate with or without watermark
        generated = generate_with_watermark(
            model, tokenizer, prompt, watermark_config, max_tokens
        )

        result = {
            "item_id": item.get("id", i),
            "regime": args.regime,
            "watermark": args.watermark,
            "strength": args.strength if args.watermark != "none" else 0.0,
            "source": source_text,
            "generated": generated,
        }

        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(passages)}", file=sys.stderr)

    # Save results
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Saved {len(results)} results to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
