"""LoRA supervised fine-tuning of Qwen3-4B-Instruct on Helix briefs."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from helix.config import ADAPTER_DIR, BASE_LLM, DOCS_DIR, MAX_SFT_LENGTH, RUNS_DIR, SFT_DIR, ensure_dirs
from helix.data.build_sft import write_datasets


class ChatDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _token_ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    return list(encoded["input_ids"])


def _encode(tokenizer, messages: list[dict]) -> dict | None:
    prompt_ids = _token_ids(tokenizer, messages[:-1], add_generation_prompt=True)
    full_ids = _token_ids(tokenizer, messages, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt is not a prefix of the full example")
    if len(full_ids) > MAX_SFT_LENGTH:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def _collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, mask = [], [], []
    for row in batch:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        mask.append(row["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def _deduped(rows: list[dict]) -> list[dict]:
    """Drop wrapper duplicates so a CPU run still sees every task and note."""
    kept = []
    for row in rows:
        content = row["messages"][1]["content"]
        if "Question: Using only the sources, " in content:
            continue
        if "Question: For the decision desk: " in content:
            continue
        kept.append(row)
    return kept


def train(
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 1e-4,
    device: str | None = None,
    dedupe: bool = False,
) -> dict:
    """Train the Helix LoRA adapter and write a model card stub with the loss."""
    ensure_dirs()
    summary = write_datasets()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_LLM)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_rows = _load_jsonl(SFT_DIR / "train.jsonl")
    if dedupe or device == "cpu":
        raw_rows = _deduped(raw_rows)
        summary = {**summary, "cpu_deduped": len(raw_rows)}
    encoded = []
    skipped = 0
    for row in raw_rows:
        item = _encode(tokenizer, row["messages"])
        if item is None:
            skipped += 1
        else:
            encoded.append(item)
    print(f"encoded {len(encoded)} skipped {skipped}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_LLM,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad], lr=lr
    )
    loader = DataLoader(
        ChatDataset(encoded),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, tokenizer.pad_token_id),
    )
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(epochs):
        for index, batch in enumerate(loader, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            if index % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                losses.append(float(loss.detach().cpu()) * grad_accum)
                if step % 10 == 0:
                    print(f"epoch {epoch + 1} step {step} loss {losses[-1]:.4f}", flush=True)
                if step % 40 == 0:
                    model.save_pretrained(ADAPTER_DIR)
                    tokenizer.save_pretrained(ADAPTER_DIR)
        print(f"finished epoch {epoch + 1}", flush=True)
        model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    report = {
        "base": str(BASE_LLM),
        "examples": summary,
        "encoded": len(encoded),
        "skipped": skipped,
        "epochs": epochs,
        "steps": step,
        "mean_loss": round(sum(losses) / max(1, len(losses)), 4),
        "lora_r": 16,
        "lr": lr,
        "device": device,
    }
    (RUNS_DIR / "sft_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_card(report)
    print(json.dumps(report, indent=2))
    return report


def _write_card(report: dict) -> None:
    card = f"""# Helix language-model card

## Model

LoRA adapter on `{report["base"]}` (Qwen3-4B-Instruct-2507). The adapter is `models/helix-lora`.

## Data

See `docs/dataset_card.md`. Encoded training examples: {report["encoded"]}. Skipped for length: {report["skipped"]}.

## Training

- Method: supervised fine-tuning, loss on assistant tokens only
- LoRA rank: {report["lora_r"]}, alpha 32, attention projections
- Epochs: {report["epochs"]}
- Optimizer steps: {report["steps"]}
- Learning rate: {report["lr"]}
- Mean loss: {report["mean_loss"]}
- Device: {report["device"]}
- Precision: bfloat16

Preference and structured-accuracy numbers are written to `docs/eval_report.md` after evaluation.

## Intended use

Grounded extraction, briefs, risk notes, quoted answers, and pairwise comparisons over a user collection. Outputs are checked against cited pages before display.

## Limitations

The adapter is specialized to this schema and this style of technical note. It is not a general replacement for the base instruct model.
"""
    (DOCS_DIR / "model_card_llm.md").write_text(card, encoding="utf-8")


def main() -> None:
    if torch.cuda.is_available():
        train()
    else:
        # The 5090 fell off the bus mid-run (Xid 79). One epoch of the
        # deduplicated set is the CPU fallback.
        train(epochs=1, batch_size=1, grad_accum=4, device="cpu", dedupe=True)


if __name__ == "__main__":
    main()
