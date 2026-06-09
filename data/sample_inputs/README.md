# Sample Input Data

This directory contains sample input data for reproducibility verification.

## Files

- `sample_candidates.jsonl`: Sample of 10 candidate records from the main dataset
- `abs_ppl_index.jsonl`: Perplexity index for corpus selection

## Full Dataset

Due to size constraints (~30GB), the full experimental data is not included in this repository.

The complete dataset contains:
- 13,600 candidates across multiple regimes (R0/R1/R1.5/R2)
- Three watermark families (KGW, UW, EXP/Gumbel)
- Multiple strength levels and variations

### Data Format

Each JSONL record contains:
- `item_id`: Source passage identifier
- `regime`: Rewrite regime (R0/R1/R1.5/R2)
- `watermark`: Watermark configuration
- `text`: Generated/rewritten text
- `metadata`: Generation parameters and statistics

## Reproduction

To reproduce the full dataset:
1. Run the generation scripts in `code/experiments/`
2. Expected runtime: 50-100 GPU hours on RTX 5090 32GB
3. Expected storage: ~30GB

Contact the authors after deanonymization for access to the full preprocessed dataset.
