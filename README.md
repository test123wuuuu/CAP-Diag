<<<<<<< HEAD
# CAP-Diag: Diagnosing Detector-Statistic-Accessible Information in LLM Watermarks under Controlled Rewrite

Anonymous submission to EMNLP 2026.

## Overview

This repository contains the code and paper for **CAP-Diag**, a diagnostic framework that measures detector-statistic-accessible mutual information in LLM watermarks under controlled rewrite transformations.

## Abstract

Watermark detectors are evaluated by AUROC/TPR/robustness benchmarks, but those metrics do not say how much information the chosen detection statistic $S$ still carries about the watermark flag $W$ once the text has been meaning-preservingly rewritten. We propose **CAP-Diag**, a diagnostic that estimates the *detector-statistic-accessible* MI $\hat{I}(W;S)$ under a controlled rewrite ladder (R0/R1/R1.5/R2), with a Fano-style chosen-statistic error floor for any classifier consuming $S$.

On 200 synthetic and 300 Wikipedia sources across three token-level watermark families (KGW, UW, EXP/Gumbel), CAP-Diag exposes a *regime/length-coupled low-information bottleneck*: source-paired R1/R0 MI ratio ≈4-15× across families and corpora.

## Repository Structure

```
anonymous-emnlp2026/
├── code/                       # Experimental code
│   ├── protocols.py           # Core rewrite regime definitions
│   └── experiments/           # Experiment scripts
│       ├── capacity/          # Capacity measurement experiments
│       ├── triage/            # CAP-Triage experiments
│       ├── ablations/         # Ablation studies
│       └── length_control/    # Length disentanglement
├── data/                       # Data samples
│   └── sample_inputs/         # Sample input data
```

## Key Contributions

1. **CAP-Diag Protocol**: A reusable diagnostic for measuring detector-statistic-accessible MI under controlled rewrites
2. **Regime Ladder**: Four rewrite regimes (R0/R1/R1.5/R2) with NLI gates and length controls
3. **Empirical Findings**: 
   - R1/R0 MI ratio of 4-15× across three watermark families
   - Length is the dominant driver with regime-specific residual
   - Low-capacity rewrite attack achieving ~80% MI drop
4. **CAP-Triage**: Abstention readout for deployment scenarios

## Requirements

See `requirements.txt` for Python dependencies.

Key dependencies:
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (tested on RTX 5090 32GB)

## Reproduction

See `REPRODUCE.md` for detailed reproduction instructions.

Quick start:
```bash
# Install dependencies
pip install -r requirements.txt

# Run sample experiment
python code/experiments/capacity/e3_generate_exp.py --help
```

## Citation

```bibtex
@inproceedings{anonymous2026capdiag,
  title={CAP-Diag: Diagnosing Detector-Statistic-Accessible Information in LLM Watermarks under Controlled Rewrite},
  author={Anonymous},
  booktitle={Proceedings of EMNLP 2026},
  year={2026}
}
```

## License

This code is released for research purposes. See LICENSE for details.



>>>>>>> 
