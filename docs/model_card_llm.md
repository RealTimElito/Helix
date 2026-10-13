# Helix language-model card

## Model

LoRA adapter on `models/Qwen3-4B-Instruct-2507` (Qwen3-4B-Instruct-2507). The adapter is `models/helix-lora`. The base weights are downloaded separately under the Qwen license.

## Data

See `docs/dataset_card.md`. Encoded training examples: 688. Skipped for length: 0.

## Training

- Method: supervised fine-tuning, loss on assistant tokens only
- LoRA rank: 16, alpha 32, attention projections
- Epochs: 1
- Optimizer steps: 172
- Learning rate: 0.0001
- Mean loss: 0.0111
- Device: cpu
- Precision: bfloat16

Preference and structured-accuracy numbers are written to `docs/eval_report.md` after evaluation.

## Intended use

Grounded extraction, briefs, risk notes, quoted answers, and pairwise comparisons over a user collection. Outputs are checked against cited pages before display.

## Limitations

The adapter is specialized to this schema and this style of technical note. It is not a general replacement for the base instruct model.
