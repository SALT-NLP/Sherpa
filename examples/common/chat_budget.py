from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from areal.utils.hf_utils import load_hf_tokenizer


def _estimate_text_tokens(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass(slots=True)
class ChatContextBudget:
    tokenizer_path: str | None = None
    context_length: int | None = None
    safety_margin: int = 256

    def count_message_tokens(self, messages: list[dict[str, Any]]) -> int:
        if self.tokenizer_path:
            tokenizer = load_hf_tokenizer(self.tokenizer_path)
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(token_ids, Mapping):
                    token_ids = token_ids["input_ids"]
                return len(token_ids)
            except Exception:
                pass

            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return len(tokenizer.encode(rendered))
            except Exception:
                pass

        fallback_text = "\n".join(
            f"[{message.get('role', 'unknown')}]\n{message.get('content', '')}"
            for message in messages
        )
        return _estimate_text_tokens(fallback_text)

    def clamp_max_completion_tokens(
        self,
        messages: list[dict[str, Any]],
        requested_max_completion_tokens: int,
    ) -> tuple[int, int]:
        prompt_tokens = self.count_message_tokens(messages)
        if self.context_length is None:
            return requested_max_completion_tokens, prompt_tokens
        available = self.context_length - self.safety_margin - prompt_tokens
        return max(
            0, min(int(requested_max_completion_tokens), available)
        ), prompt_tokens
