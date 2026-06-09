# CAP-Diag: Diagnosing Detector-Statistic-Accessible Information in LLM Watermarks

Anonymous submission to EMNLP 2026.

## Overview

This repository contains code and supplementary materials for **CAP-Diag**, a diagnostic framework that measures detector-statistic-accessible mutual information in LLM watermarks under controlled rewrite transformations.

**Paper**: See `paper/main.pdf`

## Repository Structure

```
.
├── code/                           # Core experimental code
│   ├── protocols.py                # Rewrite regime definitions (R0/R1/R2)
│   ├── generate_watermarked_text.py    # Text generation with watermarks
│   ├── compute_mi_estimates.py     # MI estimation with bootstrap CI
│   ├── length_controlled_analysis.py   # Length disentanglement experiment
│   └── triage_analysis.py          # CAP-Triage implementation
├── paper/                          # LaTeX source and PDF
│   ├── main.pdf                    # Compiled paper
│   ├── main.tex                    # Main LaTeX source
│   └── tables/                     # Table definitions
├── data/                           # Sample data
│   └── sample_inputs/              # Example input format
└── requirements.txt                # Python dependencies
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

Generate watermarked text:
```bash
python code/generate_watermarked_text.py \
    --input data/sources.jsonl \
    --output data/generated.jsonl \
    --regime R1 \
    --watermark kgw \
    --strength 2.0
```

Compute MI estimates:
```bash
python code/compute_mi_estimates.py \
    --input data/detection_stats.jsonl \
    --output results/mi_estimates.json
```

## Key Components

- **Rewrite Regimes** (R0/R1/R2): Controlled text transformations with NLI-based semantic gates
- **MI Estimation**: Clipped Miller-Madow estimator with source-paired bootstrap
- **Length Control**: Disentangles length effects from regime effects
- **CAP-Triage**: Converts MI diagnosis into abstention readout

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended: ≥24GB VRAM)

See `requirements.txt` for complete dependencies.

## Reproduction

### Main Experiments

1. **Table 1** (Main Diagnostic): Generate R0/R1/R2 texts, compute MI estimates
2. **Table 2** (Cross-Mechanism): Validate with EXP watermark
3. **Figure 2** (Length Disentanglement): Length-controlled comparison

See individual script help for detailed usage:
```bash
python code/generate_watermarked_text.py --help
```

### Data

Due to size constraints (~30GB), the full experimental dataset is not included. Sample data format is provided in `data/sample_inputs/`. Contact authors after deanonymization for access to full preprocessed data.

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

MIT License. See LICENSE file for details.

## Contact

For questions during review, please use the anonymous submission system.
