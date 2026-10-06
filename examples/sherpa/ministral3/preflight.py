"""Read-only checks for the native, text-only Ministral training recipe."""

import argparse
import importlib.metadata
import os

from transformers import AutoConfig, AutoTokenizer

from examples.sherpa.core.callers import apply_chat_template
from examples.sherpa.eval.evaluator import load_experiment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    # Only resolves the config's student endpoint; no HTTP client is created.
    os.environ.setdefault("STUDENT_BASE_URL", "http://127.0.0.1:1/v1")
    config, _ = load_experiment_config(args.config, args.overrides)
    hf_config = AutoConfig.from_pretrained(config.actor.path, local_files_only=True)
    if (
        hf_config.model_type != "mistral3"
        or hf_config.text_config.model_type != "ministral3"
    ):
        raise ValueError("Expected the official full Mistral3/Ministral3 checkpoint.")
    targets = set(config.actor.target_modules)
    expected = {
        f"model.language_model.layers.{i}.{module}"
        for i in range(hf_config.text_config.num_hidden_layers)
        for module in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    if targets != expected:
        raise ValueError(
            "LoRA targets must cover only the language layers of this checkpoint."
        )
    if (
        config.actor.fsdp.memory_efficient_load
        or config.actor.weight_update_mode != "disk"
    ):
        raise ValueError("Use native pretrained loading and disk weight updates.")
    if not config.actor.use_lora:
        raise ValueError(
            "This recipe is LoRA training, not a full-finetuning fallback."
        )
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    if not callable(flash_attn_func) or not callable(flash_attn_varlen_func):
        raise ValueError("The existing environment must provide FlashAttention 2.")
    tokenizer = AutoTokenizer.from_pretrained(config.actor.path, local_files_only=True)
    ids = apply_chat_template(
        tokenizer, [{"role": "user", "content": "Hello"}], enable_thinking=False
    )
    if not ids or not all(isinstance(i, int) for i in ids):
        raise ValueError("Chat template did not return integer token IDs.")
    if tokenizer.eos_token_id != 2:
        raise ValueError("Unexpected EOS; review config stop_token_ids.")
    for name in ("transformers", "sglang", "torch", "peft"):
        print(f"{name}: {importlib.metadata.version(name)}")
    print(
        f"Training config/tokenizer checks passed: {len(targets)} language LoRA targets."
    )
    if args.training_only:
        print(
            "This preflight does not test GPU forward/backward or rollout adapter reload."
        )
        return

    # Inspect SGLang's own config, not raw AutoConfig: its official loader copies
    # text attributes onto the outer config for the inference/LoRA interfaces.
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.lora.utils import get_normalized_target_modules

    inference_config = ModelConfig(config.actor.path)
    if (
        inference_config.hf_config.num_hidden_layers
        != hf_config.text_config.num_hidden_layers
    ):
        raise ValueError("SGLang did not resolve the text model's layer count.")
    if get_normalized_target_modules(targets) != {
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    }:
        raise ValueError("SGLang did not resolve the language LoRA targets.")
    print("Native SGLang configuration checks passed (text-only, TP=1).")
    print(
        "Run a short GPU smoke test after code/config changes; this preflight is CPU-only."
    )


if __name__ == "__main__":
    main()
