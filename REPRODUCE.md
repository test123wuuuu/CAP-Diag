# Reproduction Guide

This document provides step-by-step instructions for reproducing the experiments in the CAP-Diag paper.

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥24GB VRAM (tested on RTX 5090 32GB)
- **RAM**: ≥64GB recommended
- **Storage**: ~50GB for models and intermediate data
- **OS**: Linux or Windows with CUDA support

## Software Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv capdiag-env
source capdiag-env/bin/activate  # On Windows: capdiag-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Downloads

The experiments use the following models:
- **Qwen2.5-7B-Instruct** (text generation)
- **DeBERTa-v3-large** (NLI filtering)

```bash
# Models will be auto-downloaded on first run
# Or pre-download with:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/deberta-v3-large')"
```

## Core Experiments

### Experiment 1: Main Diagnostic (Table 1)

Generate watermarked texts across R0/R1/R1.5/R2 regimes for KGW and UW:

```bash
cd code

# Generate R0/R1/R2 candidates for KGW watermark
python experiments/capacity/measure_capacity.py \
    --regime R0 --watermark kgw --strength 5 \
    --out ../data/capacity_kgw_r0.jsonl

python experiments/capacity/measure_capacity.py \
    --regime R1 --watermark kgw --strength 5 \
    --out ../data/capacity_kgw_r1.jsonl

# Repeat for UW watermark
python experiments/capacity/measure_capacity.py \
    --regime R0 --watermark uw --strength 5 \
    --out ../data/capacity_uw_r0.jsonl

python experiments/capacity/measure_capacity.py \
    --regime R1 --watermark uw --strength 5 \
    --out ../data/capacity_uw_r1.jsonl
```

**Expected runtime**: ~20-30 GPU hours per watermark family
**Expected output**: MI estimates with bootstrap confidence intervals

### Experiment 2: EXP/Gumbel Mechanism (Table 2)

Cross-mechanism validation with EXP watermark:

```bash
python experiments/capacity/e3_generate_exp.py \
    --out ../data/candidates_exp.jsonl

python experiments/capacity/e3_capacity_exp.py \
    --candidates ../data/candidates_exp.jsonl \
    --out ../data/capacity_exp_results.json
```

**Expected runtime**: ~4-6 GPU hours
**Expected output**: R0 vs R1 MI ratio with permutation test p-value

### Experiment 3: Length Disentanglement (Figure 2)

Controlled length-matched comparison:

```bash
# Generate R1-short (length-matched to R0)
python experiments/length_control/round11_generate_r1short.py \
    --target_length 30-50 \
    --out ../data/candidates_r1short.jsonl

# Compute metrics
python experiments/length_control/round11_r1short_metrics.py \
    --r0 ../data/capacity_kgw_r0.jsonl \
    --r1short ../data/candidates_r1short.jsonl \
    --out ../data/length_disentangle_results.json
```

**Expected runtime**: ~10-15 GPU hours
**Expected output**: Source-paired R1-short/R0 ratio with CIs

### Experiment 4: CAP-Triage (Figure 4)

Risk-coverage analysis for abstention:

```bash
python experiments/triage/e2_cap_triage_sim.py \
    --capacity ../data/capacity_kgw_r0.jsonl \
    --out ../data/triage_results.json
```

**Expected runtime**: ~2-4 hours (CPU-bound)
**Expected output**: Risk-coverage curves and held-out validation

### Experiment 5: Ablations (Appendix)

```bash
python experiments/ablations/phase_f_ablations.py \
    --out ../data/ablation_results/
```

**Expected runtime**: ~5-10 GPU hours
**Expected output**: NLI threshold sensitivity, strength sweep, etc.

## Verification

To verify reproduction correctness:

1. **Check data format**: Each JSONL output should contain records with `item_id`, `regime`, `text`, `watermark`, `z_score`
2. **Spot-check statistics**: 
   - R0 mean length should be ~30-35 tokens
   - R1 mean length should be ~80-120 tokens
   - NLI contradiction rate should be <5%
3. **Compare MI estimates**: Bootstrap CIs should overlap with paper values within expected sampling variance

## Common Issues

### Out of Memory
- Reduce batch size in generation scripts (`--batch_size 1`)
- Use gradient checkpointing
- Switch to smaller model (results may differ)

### NLI Gate Too Strict
- Check DeBERTa model loaded correctly
- Verify NLI threshold (default 0.5)
- Inspect rejected samples

### MI Estimate Unstable
- Increase bootstrap iterations (`--bootstrap 500` → `1000`)
- Increase sample size (`--n_sources 200` → `300`)
- Check balanced class distribution

## Reproducing the Paper Figures

All paper figures can be regenerated from the experiment outputs:

```bash
# Figure 1: MI Ladder
python code/plot_mi_ladder.py --data data/capacity_*.jsonl

# Figure 2: Length Disentanglement  
python code/plot_length_disentangle.py --data data/length_disentangle_results.json

# Figure 3: Low-Capacity Attack
python code/plot_attack_roc.py --data data/lowcap_attack_*.jsonl

# Figure 4: CAP-Triage
python code/plot_triage.py --data data/triage_results.json
```

**Note**: Plotting scripts are not included in this minimal release. Contact authors for full plotting code.

## Timeline

Expected total reproduction time:
- Setup: 1-2 hours
- Main experiments: 50-100 GPU hours
- Analysis: 5-10 hours

Total: ~3-5 days with single RTX 5090 GPU

## Data Availability

Due to size constraints (~30GB), the full preprocessed dataset is not included in this repository. 

To obtain the full dataset for faster reproduction:
- After deanonymization, contact the authors
- Dataset will be hosted on Hugging Face Datasets

## Questions

If you encounter issues during reproduction:
1. Check the GitHub Issues (after deanonymization)
2. Contact authors through the submission system (during review)

## Reproducibility Checklist

- [ ] Environment setup complete
- [ ] Models downloaded
- [ ] E1 (Main diagnostic) runs without errors
- [ ] E2 (EXP mechanism) runs without errors
- [ ] E3 (Length control) runs without errors
- [ ] E4 (CAP-Triage) runs without errors
- [ ] E5 (Ablations) runs without errors
- [ ] Output statistics match expected ranges
- [ ] MI estimates have non-overlapping CIs for R0 vs R1
