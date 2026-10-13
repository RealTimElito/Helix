"""Local Qwen3 generation, with the Helix LoRA adapter when it exists."""

from __future__ import annotations

import threading
from pathlib import Path

import torch

from helix.config import ADAPTER_DIR, BASE_LLM

_LOCK = threading.Lock()
_SINGLETON: "LocalLLM | None" = None


class LocalLLM:
    """Greedy chat generation for Helix JSON tasks."""

    def __init__(self, base: Path | None = None, adapter: Path | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base = Path(base or BASE_LLM)
        self.adapter = adapter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base,
            dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        if adapter and Path(adapter).exists():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.adapter_name = str(adapter)
        else:
            self.adapter_name = None
        self.model.eval()

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
    ) -> list[str]:
        """Generate one completion per message list."""
        self.tokenizer.padding_side = "left"
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in batch_messages
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        texts = []
        for row in output:
            generated = row[prompt_width:]
            texts.append(self.tokenizer.decode(generated, skip_special_tokens=True).strip())
        return texts

    def generate(self, messages: list[dict], max_new_tokens: int = 700) -> str:
        return self.generate_batch([messages], max_new_tokens=max_new_tokens)[0]

    def close(self) -> None:
        del self.model
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_llm() -> LocalLLM:
    """Load the fine-tuned adapter if present, otherwise the base model."""
    global _SINGLETON
    with _LOCK:
        if _SINGLETON is None:
            adapter = ADAPTER_DIR if (ADAPTER_DIR / "adapter_config.json").exists() else None
            _SINGLETON = LocalLLM(adapter=adapter)
        return _SINGLETON


def generate_locked(messages: list[dict], max_new_tokens: int = 700) -> str:
    """Serialize generation so the API and a background thread cannot overlap."""
    llm = get_llm()
    with _LOCK:
        return llm.generate(messages, max_new_tokens=max_new_tokens)
