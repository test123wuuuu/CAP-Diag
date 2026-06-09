"""E3: Reduced sanity check with EXP/Gumbel watermark (third mechanism family).

Scope (per paper-plan):
  - Qwen2.5-7B-Instruct only
  - R0 + R1 only (the two regimes where we have the clearest contrast)
  - peak strength only (EXP strength=1.0; no_wm for baseline)
  - n=200 sources, variant A only (reduces cost 2×; capacity statistic still
    has paired-bootstrap CIs)
  - target output: ~2-6 GPU-h

Goal: confirm that R0 capacity << R1 capacity holds on a watermark whose
mechanism is fundamentally different from greenlist (KGW/UW). Success gate:
bootstrap CIs of R0 vs R1 MI do not overlap, or permutation p<0.01.

If pass: contributes to "capacity framing holds across mechanisms" claim.
If fail: paper narrows to "greenlist-family token watermarks".
"""
import argparse, json, sys, io, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
# sys.path.insert removed for clean repo structure
from exp_watermark import ExpLogitsProcessor
from protocols import R0_PROMPT, R1_PROMPT


REGIME_PROMPTS = {"R0": R0_PROMPT, "R1": R1_PROMPT}
REGIME_MAX_TOKENS = {"R0": 160, "R1": 240}


def deterministic_seed(item_id, variant, base_seed):
    import hashlib
    h = hashlib.md5(f"{item_id}|{variant}|{base_seed}|exp".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./data/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--episodes", default="./data/data/episodes.jsonl")
    ap.add_argument("--out", default="./data/data/candidates_exp.jsonl")
    ap.add_argument("--secret_key", type=int, default=20260513)
    ap.add_argument("--base_seed", type=int, default=20260513)
    ap.add_argument("--n_sources", type=int, default=200)
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(args.episodes, encoding="utf-8")][:args.n_sources]
    print(f"using {len(eps)} sources × R0+R1 × (no_wm + EXP peak) × variant A", flush=True)

    print("loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()
    vocab_size = model.config.vocab_size
    print(f"model ready; vocab={vocab_size}", flush=True)

    fout = open(args.out, "w", encoding="utf-8")
    total = 0; t_start = time.time()
    n_planned = len(eps) * 2 * 2  # variant A × (R0+R1) × (no_wm + exp)

    for ei, ep in enumerate(eps):
        variant = "A"
        src = ep[f"target_{variant}"]
        for regime in ("R0", "R1"):
            prompt = REGIME_PROMPTS[regime].format(target=src)
            max_new = REGIME_MAX_TOKENS[regime]
            msgs = [{"role": "user", "content": prompt}]
            inp = tok.apply_chat_template(msgs, return_tensors="pt",
                                          add_generation_prompt=True).to(model.device)
            attn = torch.ones_like(inp)
            seed = deterministic_seed(ep["item_id"], variant, args.base_seed)

            for family, strength in (("no_wm", 0.0), ("exp", 1.0)):
                if family == "no_wm":
                    procs = LogitsProcessorList([])
                else:
                    procs = LogitsProcessorList([
                        ExpLogitsProcessor(vocab_size=vocab_size, strength=1.0,
                                           secret_key=args.secret_key,
                                           prompt_len=inp.shape[1])
                    ])
                torch.manual_seed(seed)
                with torch.no_grad():
                    if family == "exp":
                        # EXP at strength=1.0 is deterministic given context;
                        # disable sampling so transformers respects the
                        # logits-processor's hard select.
                        out = model.generate(
                            inp, attention_mask=attn, max_new_tokens=max_new,
                            do_sample=False, logits_processor=procs,
                            pad_token_id=tok.eos_token_id,
                        )
                    else:
                        out = model.generate(
                            inp, attention_mask=attn, max_new_tokens=max_new,
                            do_sample=True, top_p=0.95, temperature=0.7,
                            logits_processor=procs,
                            pad_token_id=tok.eos_token_id,
                        )
                new_tokens = out[0][inp.shape[1]:].tolist()
                # strip trailing EOS for clean detection
                if new_tokens and new_tokens[-1] == tok.eos_token_id:
                    new_tokens = new_tokens[:-1]
                text = tok.decode(new_tokens, skip_special_tokens=True).strip()
                rec = {
                    "item_id": ep["item_id"], "domain": ep["domain"],
                    "variant": variant, "regime": regime,
                    "family": family, "strength": strength,
                    "secret_key": args.secret_key,
                    "gen_seed": seed,
                    "source_text": src, "text": text,
                    "n_chars": len(text), "n_tokens": len(new_tokens),
                    "token_ids": new_tokens,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                total += 1
        if (ei + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = total / max(elapsed, 1)
            eta = (n_planned - total) / max(rate, 1e-6) / 60
            print(f"[{ei+1}/{len(eps)}] gens={total} | {rate:.1f}/s | ETA {eta:.1f} min", flush=True)
    fout.close()
    print(f"\ndone -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
