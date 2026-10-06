"""Teacher calls to a commercial API, for configs with ``teacher.provider``.

Launched by ``eval/runner.py``; it runs the evaluator in-process with the
teacher client adapted to the provider (request conventions and usage records).
Everything else follows the evaluation protocol.

All providers default to the API evaluation protocol: medium reasoning, plain-text
thinking format, provider-controlled sampling, no seed, and no explicit teacher
output cap. Use --teacher-format config to retain the YAML response protocol
instead. Only teacher final content is parsed; internal reasoning is never student
text. --teacher-sampling config explicitly sends the YAML temperature/top_p
instead. Explicit CLI/config reasoning settings override medium; providers must
support the selected effort (there is no silent fallback). Effective settings are
included in the resume signature. By default teacher requests omit output-token
limits, including pre-solve, and disable the local training-sample token cap.
Service-side limits still apply. Use --teacher-output-limit config to restore YAML
budgets; only in that mode does --teacher-thinking-token-reserve add an allowance
for internal reasoning. teacher_usage.jsonl records returned usage and finish
reasons, including retries.

Missing keys use EMPTY for unauthenticated local servers. Remove --dry-run to
evaluate; add --resume to continue the same output directory. Extra evaluator
arguments and Hydra overrides go after --, e.g. -- --student-name NAME
--skip-preflight. Use --evaluator-help to list those options. --skip-preflight
bypasses only the models-list check, useful for compatible endpoints without
GET /models.

Transport: OpenAI-compatible chat completions only; native Gemini, Anthropic
and Responses protocols are not supported. The complete API base URL is used
verbatim; /v1 is not appended. Provider request options can be supplied through
-- --teacher-request-params-file FILE. Unsupported sampling options fail
visibly; this entrypoint never silently changes prompts or budgets.

Student and auxiliary model identities, prompts, decoding, gates, presolve,
retests, dataset expansion, result aggregation and trace format follow the YAML.
The auxiliary endpoint must serve the protocol's judge model, without a teacher
LoRA. The local YAML tokenizer remains responsible for protocol context
accounting; provider token counts may differ.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit


def prepare_provider_request(
    request,
    *,
    provider,
    effort,
    sampling,
    thinking_reserve,
    output_limit="provider-default",
):
    """Adapt transport parameters, never merge native reasoning into content."""
    request = dict(request)
    # API baselines intentionally do not seed the teacher. Apply this after
    # merging request overrides, including provider fields in extra_body.
    omitted = {"seed"}
    if sampling == "provider-default":
        omitted.update({"temperature", "top_p", "top_k", "min_p"})
    if output_limit == "provider-default":
        omitted.update({"max_tokens", "max_completion_tokens", "max_output_tokens"})
    for key in omitted:
        request.pop(key, None)
    if isinstance(request.get("extra_body"), dict):
        extra_body = {
            key: value
            for key, value in request["extra_body"].items()
            if key not in omitted
        }
        if extra_body:
            request["extra_body"] = extra_body
        else:
            request.pop("extra_body")
    if effort:
        request["reasoning_effort"] = effort
    if output_limit == "config" and "max_completion_tokens" in request:
        request["max_completion_tokens"] += thinking_reserve
        if provider == "gemini":
            request["max_tokens"] = request.pop("max_completion_tokens")
    return request


def api_url(value: str) -> str:
    """Validate an explicit endpoint without rewriting a provider's API prefix."""
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("Supply a complete http(s) API base URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("Use environment variables for credentials.")
    return value


@contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--env-file", type=Path, default=Path(__file__).resolve().parents[3] / ".env"
    )
    parser.add_argument(
        "--provider", choices=["generic", "openai", "gemini"], default="generic"
    )
    parser.add_argument("--teacher-base-url", type=api_url)
    parser.add_argument("--teacher-model")
    parser.add_argument("--student-base-url", required=True, type=api_url)
    parser.add_argument("--aux-base-url", required=True, type=api_url)
    parser.add_argument("--teacher-api-key-env")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument(
        "--teacher-format",
        choices=["auto", "config", "non_thinking", "thinking"],
        default="thinking",
    )
    parser.add_argument(
        "--teacher-sampling",
        choices=["config", "provider-default"],
        default="provider-default",
    )
    parser.add_argument(
        "--teacher-output-limit",
        choices=["provider-default", "config"],
        default="provider-default",
        help="Default: omit teacher output caps; config restores YAML budgets.",
    )
    parser.add_argument(
        "--teacher-thinking-token-reserve",
        type=int,
        default=0,
        help="Explicit extra total-output allowance for internal reasoning (default 0).",
    )
    parser.add_argument("--student-api-key-env", default="STUDENT_API_KEY")
    parser.add_argument("--aux-api-key-env", default="AUX_API_KEY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--episode-timeout-seconds", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve roles and workflow locally; no API calls or output writes.",
    )
    parser.add_argument("--evaluator-help", action="store_true")
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(options.env_file, override=False)
    prefix = {"generic": "TEACHER", "openai": "OPENAI", "gemini": "GEMINI"}[
        options.provider
    ]
    options.teacher_base_url = options.teacher_base_url or os.getenv(
        f"{prefix}_BASE_URL", ""
    )
    options.teacher_model = options.teacher_model or os.getenv(f"{prefix}_MODEL", "")
    options.teacher_api_key_env = options.teacher_api_key_env or f"{prefix}_API_KEY"
    explicit_effort = options.reasoning_effort
    options.reasoning_effort = (
        options.reasoning_effort or os.getenv(f"{prefix}_REASONING_EFFORT") or "medium"
    )
    if options.provider in {"openai", "gemini"}:
        if not os.getenv(options.teacher_api_key_env):
            parser.error(
                f"Missing credential environment variable {options.teacher_api_key_env}."
            )
    if not options.teacher_model or not options.teacher_base_url:
        parser.error(
            "Provide teacher model and base URL through arguments or the selected provider's .env entries."
        )
    options.teacher_base_url = api_url(options.teacher_base_url)
    if options.teacher_thinking_token_reserve < 0:
        parser.error("--teacher-thinking-token-reserve must be non-negative.")
    if (
        options.teacher_thinking_token_reserve
        and options.teacher_output_limit != "config"
    ):
        parser.error(
            "--teacher-thinking-token-reserve requires --teacher-output-limit config."
        )
    if (
        options.provider == "gemini"
        and options.teacher_model == "gemini-3.8-flash"
        and options.reasoning_effort not in {"low", "medium", "high"}
    ):
        parser.error("Gemini 3.8 Flash supports low, medium, high thinking.")
    extra = options.evaluator_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    reserved = {
        "--config",
        "--teacher-base-url",
        "--teacher-model",
        "--api-key",
        "--self-aux-via-teacher",
        "--output-dir",
        "--concurrency",
        "--episode-timeout-seconds",
        "--resume",
    }
    if any(arg.split("=", 1)[0] in reserved for arg in extra):
        parser.error(
            "Set role/transport/output options before --; self-aux is unsupported."
        )

    # Import lazily: --help works without loading CUDA/training dependencies.
    from examples.sherpa.eval import evaluator

    argv = [
        sys.argv[0],
        "--config",
        options.config,
        "--teacher-base-url",
        options.teacher_base_url,
        "--teacher-model",
        options.teacher_model,
        "--output-dir",
        options.output_dir,
        "--concurrency",
        str(options.concurrency),
        "--episode-timeout-seconds",
        str(options.episode_timeout_seconds),
    ]
    argv += ["--resume"] if options.resume else []
    argv += ["--help"] if options.evaluator_help else extra
    previous_argv = sys.argv
    try:
        sys.argv = argv
        args = evaluator.parse_args()
    finally:
        sys.argv = previous_argv
    args.api_key = os.environ.get(options.teacher_api_key_env) or "EMPTY"
    # Capture credentials before the temporary Hydra interpolation environment
    # below replaces the evaluation role variables.
    student_api_key = os.environ.get(options.student_api_key_env) or "EMPTY"
    aux_api_key = os.environ.get(options.aux_api_key_env) or "EMPTY"

    original_load = evaluator.load_experiment_config
    original_build = evaluator.build_eval_workflow_kwargs
    original_signature = evaluator.build_run_signature

    def load_config(config_path, overrides):
        config, _ = original_load(config_path, overrides)
        if options.teacher_format not in {"auto", "config"}:
            config.teacher_response_format = options.teacher_format
        if not explicit_effort:
            options.reasoning_effort = config.teacher_api_request_params.get(
                "reasoning_effort", options.reasoning_effort
            )
        if config.auxiliary_model.mode != "api":
            raise ValueError(
                "The evaluation config must set auxiliary_model.mode to 'api'."
            )
        config.auxiliary_model.base_url = options.aux_base_url
        config.auxiliary_model.api_key = aux_api_key
        students = [asdict(student) for student in config.student_models]
        for student in students:
            student["base_url"] = options.student_base_url
            student["api_key"] = student_api_key
        return config, students

    def build_workflow(**kwargs):
        effective = original_build(**kwargs)
        if options.teacher_output_limit == "provider-default":
            effective["max_train_sample_tokens"] = None
        return effective

    def signature(**kwargs):
        result = original_signature(**kwargs)
        result["external_teacher_entrypoint"] = {
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "transport": "openai-compatible-chat-completions",
            "auxiliary_policy": "fixed-configured-model",
            "provider": options.provider,
            "reasoning_effort": options.reasoning_effort,
            "teacher_format": kwargs["config"].teacher_response_format,
            "sampling": options.teacher_sampling,
            "seed_policy": "omit",
            "output_limit": options.teacher_output_limit,
            "thinking_token_reserve": options.teacher_thinking_token_reserve,
        }
        return result

    patches = {
        "load_experiment_config": load_config,
        "build_eval_workflow_kwargs": build_workflow,
        "build_run_signature": signature,
        "teacher_request_defaults": lambda seed: {},
        "normalize_base_url": api_url,
    }
    original_client = evaluator.ApiTeacherClient

    def teacher_client(**kwargs):
        client = original_client(**kwargs)
        original_create = client._client.chat.completions.create

        async def create(**request):
            # Apply overrides before adapting the final wire request, so request
            # params cannot reintroduce a removed output limit.
            request = evaluator.merge_dicts(client.request_params, request)
            request["model"] = client.model
            request = prepare_provider_request(
                request,
                provider=options.provider,
                effort=options.reasoning_effort,
                sampling=options.teacher_sampling,
                thinking_reserve=options.teacher_thinking_token_reserve,
                output_limit=options.teacher_output_limit,
            )
            response = await original_create(**request)
            # Record provider token usage (including internal reasoning) separately
            # from student-visible traces. Never record credentials or request bodies.
            usage = getattr(response, "usage", None)
            if usage is not None:
                usage = (
                    usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
                )
            record = {
                "model": options.teacher_model,
                "usage": usage,
                "finish_reason": response.choices[0].finish_reason
                if response.choices
                else None,
            }
            with (Path(options.output_dir) / "teacher_usage.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            # ExternalActorCaller reads only message.content, not reasoning_content.
            return response

        client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        return client

    patches["ApiTeacherClient"] = teacher_client
    originals = {key: getattr(evaluator, key) for key in patches}
    # Configuration interpolation only; no shell sourcing or YAML copy.
    env = {
        "EVAL_STUDENT_URL": options.student_base_url,
        "EVAL_STUDENT_KEY": student_api_key,
        "EVAL_AUX_URL": options.aux_base_url,
        "EVAL_AUX_KEY": aux_api_key,
        # The evaluator prefers TEACHER_API_KEY over --api-key; the key selected
        # by --teacher-api-key-env has already been passed explicitly.
        "TEACHER_API_KEY": "",
    }
    try:
        for key, value in patches.items():
            setattr(evaluator, key, value)
        with environment(env):
            if options.dry_run:
                config, students = load_config(args.config, args.overrides)
                evaluator.tutor_train._apply_eval_average_rollouts(config)
                evaluator.resolve_teacher_generation_args(args, config)
                evaluator.validate_args(args)
                students = evaluator.select_student_models(students, args.student_name)
                modes = evaluator.resolve_presolve_modes(
                    args.teacher_presolve,
                    evaluator.effective_eval_presolve_enabled(config),
                )
                workflow = build_workflow(
                    config=config,
                    student_models=students,
                    tokenizer=None,
                    args=args,
                    presolve_enabled=modes[0].enabled,
                )
                print(
                    json.dumps(
                        {
                            "teacher": {
                                "model": args.teacher_model,
                                "url": options.teacher_base_url,
                            },
                            "students": [
                                {
                                    "name": s["name"],
                                    "model": s["model"],
                                    "url": s["base_url"],
                                }
                                for s in students
                            ],
                            "auxiliary": {
                                "model": workflow["aux_model"],
                                "url": workflow["aux_base_url"],
                            },
                            "temperature": (
                                args.teacher_temperature
                                if options.teacher_sampling == "config"
                                else None
                            ),
                            "seed_policy": "omit",
                            "max_tokens": (
                                args.teacher_max_tokens
                                if options.teacher_output_limit == "config"
                                else None
                            ),
                            "teacher_output_limit": options.teacher_output_limit,
                            "max_train_sample_tokens": workflow[
                                "max_train_sample_tokens"
                            ],
                            "reasoning_effort": options.reasoning_effort,
                            "teacher_format": config.teacher_response_format,
                            "teacher_sampling": options.teacher_sampling,
                            "thinking_token_reserve": options.teacher_thinking_token_reserve,
                            "presolve_enabled": workflow["teacher_pre_enabled"],
                            "presolve_modes": [mode.name for mode in modes],
                            "presolve_verify": workflow["teacher_pre_verify"],
                            "output_dir": options.output_dir,
                            "dry_run": "No API calls or output files; dataset and connectivity not checked.",
                        },
                        indent=2,
                    )
                )
            else:
                asyncio.run(evaluator.main_async(args))
    finally:
        for key, value in originals.items():
            setattr(evaluator, key, value)


if __name__ == "__main__":
    main()
