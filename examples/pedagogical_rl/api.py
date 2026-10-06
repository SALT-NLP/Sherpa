from __future__ import annotations

import asyncio
from typing import Any, Protocol

from examples.pedagogical_rl.config import PedagogicalAPIModelConfig
from examples.sherpa.core.callers import (
    AReaLEngineAuxiliaryCaller,
    AReaLEngineChatCaller,
)

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - lightweight CPU test environments
    AsyncOpenAI = None


class ChatCompletionClient(Protocol):
    chat: Any


class PedagogicalAPIClient:
    """Small async client supporting the native multi-choice student call."""

    def __init__(
        self,
        config: PedagogicalAPIModelConfig,
        *,
        client: ChatCompletionClient | None = None,
    ) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        if client is not None:
            self.client = client
        else:
            if AsyncOpenAI is None:
                raise RuntimeError("openai is required for PedagogicalRL API calls")
            if not config.base_url:
                raise ValueError("PedagogicalRL API base_url is required")
            if not config.api_key:
                raise ValueError("PedagogicalRL API api_key is required")
            self.client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        if n < 1:
            raise ValueError("n must be positive")
        # config.timeout covers one completion; scale it by the number of choices.
        request_timeout = float(self.config.timeout) * max(1, int(n))
        seed = {} if self.config.seed is None else {"seed": self.config.seed}
        async with self._semaphore:
            response = await self.client.chat.completions.create(
                timeout=request_timeout,
                model=self.config.model,
                messages=messages,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **seed,
                extra_headers=self.config.extra_headers or None,
                extra_body={
                    "top_k": self.config.top_k,
                    "min_p": self.config.min_p,
                    "chat_template_kwargs": self.config.chat_template_kwargs,
                },
            )
        choices = sorted(response.choices, key=lambda choice: choice.index)
        if len(choices) != n:
            raise RuntimeError(
                f"API model {self.config.model!r} returned {len(choices)} choices, "
                f"expected {n}"
            )
        return [choice.message.content or "" for choice in choices]


class PedagogicalEngineClient:
    """Pedagogical model client backed by the rollout engine's base model.

    Its public ``generate`` contract matches :class:`PedagogicalAPIClient`, but
    every request carries ``disable_lora=True`` through the shared tutor engine
    caller. Judge samples therefore come from the frozen base model and never
    enter the trainable teacher interaction cache.
    """

    def __init__(
        self,
        config: PedagogicalAPIModelConfig,
        *,
        engine: Any,
        tokenizer: Any,
        base_gconfig: Any,
    ) -> None:
        self.config = config
        self.base_gconfig = base_gconfig
        self._semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self._chat_caller = AReaLEngineChatCaller(
            engine=engine,
            tokenizer=tokenizer,
            enable_thinking=bool(
                config.chat_template_kwargs.get("enable_thinking", False)
            ),
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        if n < 1:
            raise ValueError("n must be positive")
        request_gconfig = self.base_gconfig
        # AReaL's in-engine GenerationHyperparameters has no min_p field, so only
        # the backend default min_p=0 is supported.
        if self.config.min_p != 0.0:
            raise ValueError("the in-engine judge supports only min_p=0")
        if hasattr(request_gconfig, "new"):
            request_gconfig = request_gconfig.new(top_k=self.config.top_k)

        caller = AReaLEngineAuxiliaryCaller(
            chat_caller=self._chat_caller,
            base_gconfig=request_gconfig,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_concurrency=self.config.max_concurrent_calls,
            context_length=getattr(self.base_gconfig, "max_tokens", None),
            context_window_margin=0,
            semaphore=self._semaphore,
        )
        results = await caller.call_text_many(
            messages,
            n=n,
            rid_prefix="pedagogical-judge",
            timeout=float(self.config.timeout) * max(1, int(n)),
        )
        errors = [result.error for result in results if result.error]
        if errors:
            raise RuntimeError(errors[0])
        return [result.text for result in results]
