from __future__ import annotations

import re


def strip_reasoning_for_context(text: str) -> str:
    text = text or ""
    text = re.sub(
        r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<think(?:ing)?\b[^>]*>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"</?think(?:ing)?\b[^>]*>", "", text, flags=re.IGNORECASE).strip()
