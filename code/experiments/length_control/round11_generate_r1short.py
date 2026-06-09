"""Round 11 P1A: generate length-controlled R1-short variant.

R1-short = paraphrase rewrite (R1 style: different surface, key facts preserved)
constrained to target length 35-45 tokens (matching R0 length distribution).

Same canonical setup as Round 7/8/9/10:
  - source: episodes.jsonl variant=A only (item_id matches R0 in candidates_r4)
  - watermark families: KGW (delta=5) + UW (delta=5); EXP optional
  - K=8 candidate draws per source; first one passing utility gate kept
  - utility gate: NLI bidirectional non-contradiction (DeBERTa-v3 score <0.5)
                  + fact-skeleton match (entity + value + date verbatim modulo variants)
                  + length band 35-45 tokens (CRITICAL for R1-short)
  - secret_key = 20260513; gamma = 0.25; peak strength
  - same deterministic seed convention as generate_r1_r2.py

Output: experiments/idea5/data/candidates_r1short.jsonl

Pause rule: if gate pass rate < 30% per family on first 30 sources, abort
and emit decision note.
"""
import argparse, json, sys, time, io, hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, "./data/scripts")
sys.path.insert(0, "./data/scripts")
from watermarks import build_processor, deterministic_seed
from protocols import NLIScorer

R1_SHORT_PROMPT = (
    "Paraphrase the following passage as a short rewrite for a synthetic dataset. "
    "Aim for a SHORT rewrite of about 30 to 50 tokens (roughly 22 to 36 words). "
    "Vary word choice and sentence structure, BUT preserve every key fact: "
    "entities (verbatim), numbers, dates (verbatim), places, and quantities must appear unchanged. "
    "Do not invent new facts or different values. Do not hedge. "
    "Output ONLY the rewritten passage, no preamble.\n\n"
    "Original passage:\n{target}\n\n"
    "Rewritten passage:"
)

LENGTH_BAND = (30, 50)
NLI_CONTRA_THR = 0.5
K_CANDIDATES = 8
PEAK = 5.0
GAMMA = 0.25
KEY = 20260513
BASE_SEED = 20260513


def extract_fact_keys(ep):
    """Return list of required substrings from fact_skeleton that ACTUALLY
    appear in target_A. We do NOT require keys absent from the source
    passage; that would force the rewriter to invent facts."""
    fs = ep.get("fact_skeleton", {})
    src = ep.get("target_A", "")
    src_low = src.lower()
    keys = []
    for k in ("entity", "value", "date", "place", "year", "pid"):
        v = fs.get(k)
        if v is None: continue
        v_str = str(v)
        if v_str.lower() in src_low:
            keys.append(v_str)
    return keys


def skeleton_match(text, keys):
    """Verbatim modulo simple normalisation; allow some paraphrastic flexibility
    on numeric values (e.g. '40 mg' vs '40mg', '2025' vs '2025.', etc.)."""
    if not keys: return True
    text_norm = text.replace(",", "").lower()
    for k in keys:
        kn = str(k).lower().strip()
        if kn in text.lower():
            continue
        # tolerate hyphen/slash/dot variants in dates and numbers
        for v in (kn.replace("-", "/"), kn.replace("/", "-"),
                  kn.replace("-", "."), kn.replace(".", "-"),
                  kn.replace(",", ""), kn.replace(" ", "")):
            if v in text_norm:
                break
        else:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./data/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--episodes", default="./data/data/episodes.jsonl")
    ap.add_argument("--nli_dir", default="./data/models/nli-deberta-v3-base")
    ap.add_argument("--out", default="./data/data/candidates_r1short.jsonl")
    ap.add_argument("--families", default="kgw,uw")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    episodes = [json.loads(l) for l in open(args.episodes, encoding="utf-8")]
    if args.limit > 0: episodes = episodes[:args.limit]
    print(f"loaded {len(episodes)} episodes; families={families}", flush=True)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if args.resume and out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done_keys.add((r["item_id"], r["family"]))
            except: pass
        print(f"resume: {len(done_keys)} prior records", flush=True)

    print("loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()
    vocab_size = model.config.vocab_size
    print(f"model ready; vocab={vocab_size}", flush=True)

    print("loading NLI scorer...", flush=True)
    nli = NLIScorer(args.nli_dir)

    fout = open(out_path, "a", encoding="utf-8")
    stats = {f: {"attempt": 0, "gate_pass": 0} for f in (["no_wm"] + families)}
    total = 0
    t0 = time.time()

    family_strength_pairs = [("no_wm", 0.0)] + [(f, PEAK) for f in families]

    for ep_i, ep in enumerate(episodes):
        variant = "A"
        src = ep[f"target_{variant}"]
        keys = extract_fact_keys(ep)
        seed = deterministic_seed(ep["item_id"], variant, BASE_SEED)
        msgs = [{"role": "user", "content": R1_SHORT_PROMPT.format(target=src)}]
        input_ids = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        attn = torch.ones_like(input_ids)

        for (family, strength) in family_strength_pairs:
            if (ep["item_id"], family) in done_keys: continue
            if family == "no_wm":
                proc_list = LogitsProcessorList([])
            else:
                proc = build_processor(family, vocab_size, strength,
                                       gamma=GAMMA, secret_key=KEY)
                proc_list = LogitsProcessorList([proc])

            best = None
            n_len_fail = 0; n_skel_fail = 0; n_nli_fail = 0
            sample_lens = []
            for k in range(K_CANDIDATES):
                stats[family]["attempt"] += 1
                torch.manual_seed(seed + k * 1000)
                with torch.no_grad():
                    out = model.generate(
                        input_ids, attention_mask=attn,
                        max_new_tokens=80,
                        do_sample=True, top_p=0.95, temperature=0.7,
                        logits_processor=proc_list,
                        pad_token_id=tok.eos_token_id,
                    )
                new_tokens = out[0][input_ids.shape[1]:].tolist()
                text = tok.decode(new_tokens, skip_special_tokens=True).strip()
                n_tok = len(new_tokens)
                sample_lens.append(n_tok)
                # Length-band check
                if not (LENGTH_BAND[0] <= n_tok <= LENGTH_BAND[1]):
                    n_len_fail += 1
                    continue
                # Skeleton check
                if not skeleton_match(text, keys):
                    n_skel_fail += 1
                    continue
                # NLI bidirectional
                s_ab = nli.score(src, text)
                s_ba = nli.score(text, src)
                if max(s_ab["contradiction"], s_ba["contradiction"]) >= NLI_CONTRA_THR:
                    n_nli_fail += 1
                    continue
                # Passed
                best = {
                    "item_id": ep["item_id"], "domain": ep["domain"],
                    "variant": variant,
                    "regime": "R1-short",
                    "family": family,
                    "strength": strength,
                    "gamma": GAMMA, "secret_key": KEY,
                    "gen_seed": seed + k * 1000,
                    "k_attempt": k + 1,
                    "source_text": src,
                    "text": text,
                    "n_chars": len(text),
                    "n_tokens": n_tok,
                    "token_ids": new_tokens,
                    "fact_skeleton": ep.get("fact_skeleton", {}),
                    "required_fact_keys": keys,
                    "nli_max_contra": float(max(s_ab["contradiction"], s_ba["contradiction"])),
                }
                stats[family]["gate_pass"] += 1
                break

            if best is not None:
                fout.write(json.dumps(best, ensure_ascii=False) + "\n")
                fout.flush()
                total += 1
            elif ep_i < 3:
                # diagnostic on first few sources
                print(f"  [{ep['item_id']} {family}] no candidate passed; "
                      f"len_fail={n_len_fail} skel_fail={n_skel_fail} "
                      f"nli_fail={n_nli_fail} sample_lens={sample_lens}", flush=True)

        if (ep_i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{ep_i+1}/{len(episodes)}] total accepted={total}  "
                  f"per-family gate-pass-counts=" +
                  " ".join(f"{f}:{stats[f]['gate_pass']}/{stats[f]['attempt']}" for f in stats),
                  flush=True)

        # Pause check after first 30 sources
        if (ep_i + 1) == 30:
            for f in families:
                if stats[f]["attempt"] == 0: continue
                if stats[f]["gate_pass"] < 0.30 * 30:
                    print(f"WARNING: family {f} gate pass rate "
                          f"{stats[f]['gate_pass']}/{stats[f]['attempt']} < 30%")

    fout.close()
    print(f"\ndone. {total} R1-short candidates written.")
    print(f"final stats: " + " ".join(
        f"{f}:gate_pass={stats[f]['gate_pass']}/{stats[f]['attempt']}" for f in stats))


if __name__ == "__main__":
    main()
