import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.lora import inject_lora


DEFAULT_SAMPLE_PROMPTS = [
    "Instruction: Explain LoRA in one short paragraph.\nResponse:",
    "Instruction: Give two reasons why low-rank adaptation is useful for fine-tuning.\nResponse:",
    "Instruction: Summarize the difference between full fine-tuning and LoRA.\nResponse:",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def configure_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return configure_tokenizer(tokenizer)


def build_model(
    model_id: str,
    *,
    use_lora: bool,
    r: Optional[int] = None,
    alpha: Optional[int] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)

    if use_lora:
        for param in model.parameters():
            param.requires_grad = False
        model = inject_lora(model, target_modules=("q_proj", "v_proj"), r=r or 8, alpha=alpha or 16)

    return model


def parameter_statistics(model) -> Dict[str, float]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    trainable_ratio = 100.0 * trainable_params / total_params
    frozen_params = total_params - trainable_params
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(frozen_params),
        "trainable_ratio": float(trainable_ratio),
    }


def alpha_for_rank(rank: int, scaling: int = 2) -> int:
    return rank * scaling


def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def token_count_summary(dataset, field: str) -> Dict[str, float]:
    lengths = [len(row[field]) for row in dataset]
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def _tokenize_prompt_response(prompt: str, response: str, tokenizer, max_length: int) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    eos_suffix = tokenizer.eos_token or ""
    response_text = response + eos_suffix if eos_suffix and not response.endswith(eos_suffix) else response
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    # Reserve label space for the response so evaluation does not end up with all -100 labels.
    min_response_tokens = min(len(response_ids), max(16, max_length // 4))
    prompt_budget = max_length - min_response_tokens
    prompt_ids = prompt_ids[: max(prompt_budget, 0)]
    response_ids = response_ids[: max_length - len(prompt_ids)]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids.copy()
    attention_mask = [1] * len(input_ids)

    pad_length = max_length - len(input_ids)
    if pad_length > 0:
        input_ids += [tokenizer.pad_token_id] * pad_length
        labels += [-100] * pad_length
        attention_mask += [0] * pad_length

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def tokenize_instruction_batch(examples, tokenizer, max_length: int) -> Dict[str, List[List[int]]]:
    encoded = {"input_ids": [], "labels": [], "attention_mask": []}

    for prompt, response in zip(examples["prompt"], examples["response"]):
        item = _tokenize_prompt_response(prompt, response, tokenizer, max_length)
        for key in encoded:
            encoded[key].append(item[key])

    return encoded


def load_and_prepare_datasets(
    data_path: str,
    tokenizer,
    *,
    max_length: int,
    validation_size: float = 0.1,
    seed: int = 42,
) -> Tuple[DatasetDict, DatasetDict]:
    raw_dataset = load_dataset("json", data_files=data_path)["train"]
    raw_splits = raw_dataset.train_test_split(test_size=validation_size, seed=seed)
    tokenized_splits = raw_splits.map(
        lambda batch: tokenize_instruction_batch(batch, tokenizer, max_length=max_length),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc=f"Tokenizing with max_length={max_length}",
    )

    tokenized_splits = tokenized_splits.filter(
        lambda example: any(label != -100 for label in example["labels"]),
        desc="Filtering examples with no supervised tokens",
    )
    return raw_splits, tokenized_splits


def safe_perplexity(loss_value: float) -> float:
    return math.exp(min(loss_value, 20))


def generate_samples(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    device: torch.device,
    max_new_tokens: int = 80,
) -> List[Dict[str, str]]:
    model.eval()
    outputs = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append({"prompt": prompt, "generated_text": decoded})

    return outputs


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def list_result_summaries(results_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for summary_path in sorted(results_dir.glob("*/metrics_summary.json")):
        payload = load_json(summary_path)
        payload["summary_path"] = str(summary_path)
        records.append(payload)
    return records
