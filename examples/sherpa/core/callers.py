from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from examples.common.chat_budget import ChatContextBudget
from examples.common.openai_utils import AsyncLLMCaller
from examples.sherpa.core.generation_budget import (
    ContextBudgetLimitExceeded,
    ensure_response_within_train_sample_budget,
    prepare_train_sample_generation_config,
    raise_if_over_budget,
    with_max_new_tokens,
)
from examples.sherpa.core.text import strip_reasoning_for_context

from areal.api import ModelRequest, ModelResponse
from areal.api.cli_args import GenerationHyperparameters


@dataclass(slots=True)
class TextCallResult:
    text: str
    raw_text: str = ""
    error: str | None = None


@dataclass(slots=True)
class ActorCallResult:
    response: ModelResponse
    raw_text: str
    visible_text: str


@dataclass(slots=True)
class EngineChatResult:
    response: ModelResponse
    raw_text: str
    visible_text: str


def apply_chat_template(
    tokenizer: Any | None,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool = True,
) -> list[int]:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        # Transformers 5 tokenizers can return a BatchEncoding by default.
        # Iterating that mapping yields field names, not token IDs.
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        return list(encoded)
    text = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in messages
    )
    return encode_text(tokenizer, text)


def decode_output(
    tokenizer: Any | None,
    response: ModelResponse,
) -> str:
    tokenizer = response.tokenizer or tokenizer
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            # skip_special_tokens removes every model's turn terminator (e.g.
            # <|im_end|>, <|endoftext|>); the contract tags are ordinary text.
            return tokenizer.decode(response.output_tokens, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(response.output_tokens).replace("<|im_end|>", "")
    return "".join(chr(max(0, int(token))) for token in response.output_tokens)


def encode_text(tokenizer: Any | None, text: str) -> list[int]:
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    return [ord(ch) for ch in text]


class ApiAuxiliaryCaller:
    def __init__(
        self,
        caller: AsyncLLMCaller,
        *,
        request_overrides: dict[str, Any] | None = None,
    ):
        self.caller = caller
        self.request_overrides = dict(request_overrides or {})

    @property
    def request_config(self) -> dict[str, Any]:
        return {**self.caller.request_config, **self.request_overrides}

    async def call_text(
        self,
        messages: list[dict[str, str]],
        *,
        rid_prefix: str = "auxiliary",
    ) -> TextCallResult:
        del rid_prefix
        try:
            result = await self.caller.call(
                messages,
                request_overrides=self.request_overrides,
            )
        except Exception as exc:
            return TextCallResult(text="", raw_text="", error=str(exc))
        return TextCallResult(
            text=strip_reasoning_for_context(result.text),
            raw_text=result.text,
            error=None,
        )


class AReaLEngineChatCaller:
    def __init__(
        self,
        *,
        engine: Any,
        tokenizer: Any | None,
        enable_thinking: bool,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        gconfig: Any | None,
        max_completion_tokens: int,
        max_train_sample_tokens: int | None,
        metadata: dict[str, Any],
        rid_prefix: str,
    ) -> EngineChatResult:
        input_ids = apply_chat_template(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
        )
        budget = prepare_train_sample_generation_config(
            input_ids=input_ids,
            gconfig=gconfig,
            max_completion_tokens=max_completion_tokens,
            max_train_sample_tokens=max_train_sample_tokens,
        )
        raise_if_over_budget(budget)
        req = ModelRequest(
            rid=f"{rid_prefix}-{uuid.uuid4().hex}",
            input_ids=input_ids,
            gconfig=budget.gconfig,
            metadata=dict(metadata),
            tokenizer=self.tokenizer,
        )
        response = await self.engine.agenerate(req)
        ensure_response_within_train_sample_budget(
            input_len=response.input_len,
            output_len=response.output_len,
            max_train_sample_tokens=max_train_sample_tokens,
        )
        raw_text = decode_output(self.tokenizer, response)
        return EngineChatResult(
            response=response,
            raw_text=raw_text,
            visible_text=strip_reasoning_for_context(raw_text),
        )


class AReaLEngineAuxiliaryCaller:
    def __init__(
        self,
        *,
        chat_caller: AReaLEngineChatCaller,
        base_gconfig: Any | None,
        max_completion_tokens: int,
        temperature: float,
        top_p: float | None,
        max_concurrency: int,
        context_length: int | None,
        context_window_margin: int,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.chat_caller = chat_caller
        self.base_gconfig = base_gconfig
        self.max_completion_tokens = max(1, int(max_completion_tokens))
        self.temperature = float(temperature)
        self.top_p = 1.0 if top_p is None else float(top_p)
        self.context_length = context_length
        self.context_window_margin = max(0, int(context_window_margin))
        self._semaphore = semaphore or asyncio.Semaphore(max(1, int(max_concurrency)))

    async def call_text(
        self,
        messages: list[dict[str, str]],
        *,
        rid_prefix: str = "auxiliary",
    ) -> TextCallResult:
        try:
            async with self._semaphore:
                result = await self.chat_caller.generate(
                    messages,
                    gconfig=self._generation_config(),
                    metadata={"disable_lora": True},
                    max_completion_tokens=self.max_completion_tokens,
                    max_train_sample_tokens=None,
                    rid_prefix=rid_prefix,
                )
        except Exception as exc:
            return TextCallResult(text="", raw_text="", error=str(exc))
        return TextCallResult(
            text=result.visible_text,
            raw_text=result.raw_text,
            error=None,
        )

    async def call_text_many(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        rid_prefix: str = "auxiliary",
        timeout: float | None = None,
    ) -> list[TextCallResult]:
        """Sample ``n`` times. The engine backend seeds per request, so unlike
        the API caller these are separate calls rather than one ``n``-choice
        request."""

        del timeout
        return list(
            await asyncio.gather(
                *(
                    self.call_text(messages, rid_prefix=f"{rid_prefix}-{index}")
                    for index in range(n)
                )
            )
        )

    def _generation_config(self) -> Any:
        base_gconfig = self.base_gconfig or GenerationHyperparameters()
        max_tokens = getattr(base_gconfig, "max_tokens", None)
        if self.context_length is not None:
            max_tokens = max(1, int(self.context_length) - self.context_window_margin)
        kwargs = {
            "n_samples": 1,
            "max_new_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        if hasattr(base_gconfig, "new"):
            return base_gconfig.new(**kwargs)
        values = dict(getattr(base_gconfig, "__dict__", {}))
        values.update(kwargs)
        return GenerationHyperparameters(**values)


class AReaLEngineActorCaller:
    def __init__(
        self,
        *,
        chat_caller: AReaLEngineChatCaller,
        gconfig: Any | None,
        max_completion_tokens: int,
        max_train_sample_tokens: int | None,
    ) -> None:
        self.chat_caller = chat_caller
        self.gconfig = gconfig
        self.max_completion_tokens = max(1, int(max_completion_tokens))
        self.max_train_sample_tokens = max_train_sample_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        lora_version: int | None,
        rid_prefix: str,
        max_completion_tokens: int | None = None,
    ) -> ActorCallResult:
        metadata: dict[str, Any] = {}
        if lora_version is not None:
            metadata["lora_version"] = int(lora_version)
        completion_budget = (
            self.max_completion_tokens
            if max_completion_tokens is None
            else max(1, int(max_completion_tokens))
        )
        train_sample_budget = self.max_train_sample_tokens
        # The budget resolver prefers gconfig.max_new_tokens, so widen the gconfig
        # when this call asks for a larger budget; smaller requests keep the cap.
        gconfig = self.gconfig
        if gconfig is not None:
            shared = getattr(gconfig, "max_new_tokens", None)
            if shared is not None and completion_budget > int(shared):
                gconfig = with_max_new_tokens(gconfig, completion_budget)
        result = await self.chat_caller.generate(
            messages,
            gconfig=gconfig,
            max_completion_tokens=completion_budget,
            max_train_sample_tokens=train_sample_budget,
            metadata=metadata,
            rid_prefix=rid_prefix,
        )
        return ActorCallResult(
            response=result.response,
            raw_text=result.raw_text,
            visible_text=result.visible_text,
        )


class ExternalActorCaller:
    def __init__(
        self,
        *,
        client: Any,
        tokenizer: Any | None,
        context_budget: ChatContextBudget,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
        max_completion_tokens: int,
        max_train_sample_tokens: int | None,
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer
        self.context_budget = context_budget
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.enable_thinking = enable_thinking
        self.max_completion_tokens = max(1, int(max_completion_tokens))
        self.max_train_sample_tokens = max_train_sample_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        lora_version: int | None,
        rid_prefix: str,
        max_completion_tokens: int | None = None,
    ) -> ActorCallResult:
        del lora_version, rid_prefix
        completion_budget = (
            self.max_completion_tokens
            if max_completion_tokens is None
            else max(1, int(max_completion_tokens))
        )
        train_sample_budget = self.max_train_sample_tokens
        input_ids = apply_chat_template(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
        )
        budget = prepare_train_sample_generation_config(
            input_ids=input_ids,
            gconfig=None,
            max_completion_tokens=completion_budget,
            max_train_sample_tokens=train_sample_budget,
        )
        raise_if_over_budget(budget)
        safe_max_completion_tokens, _ = self.context_budget.clamp_max_completion_tokens(
            messages,
            int(budget.max_new_tokens),
        )
        if safe_max_completion_tokens <= 0:
            raise ContextBudgetLimitExceeded(
                "tutor prompt exceeded external client context budget"
            )
        response_obj = await self.client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=max(1, safe_max_completion_tokens),
        )
        raw_text = response_obj.choices[0].message.content or ""
        output_tokens = encode_text(self.tokenizer, raw_text)
        ensure_response_within_train_sample_budget(
            input_len=len(input_ids),
            output_len=len(output_tokens),
            max_train_sample_tokens=train_sample_budget,
        )
        response = ModelResponse(
            input_tokens=list(input_ids),
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            tokenizer=self.tokenizer,
        )
        return ActorCallResult(
            response=response,
            raw_text=raw_text,
            visible_text=strip_reasoning_for_context(raw_text),
        )
