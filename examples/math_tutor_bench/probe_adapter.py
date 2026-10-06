#!/usr/bin/env python3
"""Fail early if an SGLang endpoint is not actually applying the LoRA."""

from __future__ import annotations

import argparse
import os

from openai import OpenAI

MESSAGES = [
    {
        "role": "user",
        "content": (
            "You are a math teacher. In one short turn, help a student begin solving "
            "2x + 3 = 11 without merely stating the final answer."
        ),
    }
]


def chat_complete(client: OpenAI, model: str, lora_path: str | None) -> str:
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    if lora_path is not None:
        extra_body["lora_path"] = lora_path
    response = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        temperature=0.0,
        max_tokens=256,
        seed=42,
        extra_body=extra_body,
    )
    return (response.choices[0].message.content or "").strip()


def text_complete(client: OpenAI, model: str, lora_path: str | None) -> str:
    extra_body = {} if lora_path is None else {"lora_path": lora_path}
    response = client.completions.create(
        model=model,
        prompt=(
            "You are a math teacher. Help a student begin solving 2x + 3 = 11 "
            "without merely stating the final answer.\n<think>\n\n</think>\n\n"
        ),
        temperature=0.0,
        max_tokens=256,
        seed=42,
        extra_body=extra_body or None,
    )
    return (response.choices[0].text or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--api-key", default=os.environ.get("MTB_API_KEY", "EMPTY"))
    parser.add_argument(
        "--warn-if-inactive",
        action="store_true",
        help="Warn instead of failing when the adapter leaves the probe replies unchanged.",
    )
    args = parser.parse_args()

    def inactive(endpoint: str) -> None:
        message = f"LoRA is not observably active on the {endpoint} endpoint"
        if not args.warn_if_inactive:
            raise SystemExit(message)
        print(
            f"[liveness] warning: {message}; the adapter path was accepted, continuing."
        )

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=300.0,
        max_retries=0,
    )
    base_chat = chat_complete(client, args.model, None)
    try:
        chat_complete(client, args.model, "/nonexistent/mathtutorbench-adapter")
    except Exception:
        pass
    else:
        raise SystemExit("endpoint accepted a nonexistent lora_path")
    adapted_chat = chat_complete(client, args.model, args.lora_path)
    if not adapted_chat or adapted_chat == base_chat:
        inactive("chat")
        return

    base_text = text_complete(client, args.model, None)
    adapted_text = text_complete(client, args.model, args.lora_path)
    if not adapted_text or adapted_text == base_text:
        inactive("completion")
        return
    print("[liveness] LoRA is active on chat and completion endpoints")


if __name__ == "__main__":
    main()
