"""Evaluate one teacher under the paper's student-archetype protocol.

A config in configs/ names the teacher, the student and the judge, and how each
is called: an OpenAI-compatible base URL and key (both read from environment
variables) and a served model name. The teacher is either a model you serve
yourself, optionally with a LoRA adapter, or a commercial API (``provider``),
which adds that provider's request conventions. This script starts no model
server.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
PREFERENCES = (
    "none",
    "attempt-diagnosis",
    "subgoal-decomposition",
    "contrastive-comparison",
    "causal-justification",
    "step-demonstration",
    "independent-verification",
)
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def load_config(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic experiment config inheritance")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Experiment YAML must contain a mapping")
    parent = data.pop("extends", None)
    if "protocol" in data:
        data["protocol"] = str((path.parent / data["protocol"]).resolve())
    result = merge(
        load_config(path.parent / parent, (*seen, path)) if parent else {}, data
    )
    if not seen:
        validate_config(result)
    return result


def _check_fields(name: str, value, required: set, optional: set = frozenset()):
    if not isinstance(value, dict) or not required <= set(value) <= required | optional:
        raise ValueError(f"Missing or unknown fields in {name}")


def validate_config(config: dict) -> None:
    _check_fields(
        "root",
        config,
        {
            "version",
            "protocol",
            "run_name",
            "teacher",
            "roles",
            "evaluation",
            "execution",
        },
    )
    teacher = config["teacher"]
    _check_fields(
        "teacher",
        teacher,
        {
            "model",
            "endpoint_env",
            "key_env",
            "checkpoint",
            "adapter_env",
            "tokenizer",
            "format",
            "enable_thinking",
            "request_params",
        },
        {"presolve", "provider"},
    )
    _check_fields("roles", config["roles"], {"student", "judge"})
    _check_fields(
        "evaluation",
        config["evaluation"],
        {"preferences", "id_preferences", "attempts", "expected_questions"},
    )
    provider = teacher.get("provider")
    _check_fields(
        "execution",
        config["execution"],
        {
            "concurrency",
            "student_concurrency",
            "judge_concurrency",
            "episode_timeout_seconds",
            "episode_error_retries",
            "proxy",
        },
    )
    if config["version"] != 1 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["run_name"]
    ):
        raise ValueError("Require schema version 1 and a safe run_name")
    if "presolve" in teacher and type(teacher["presolve"]) is not bool:
        raise ValueError("teacher.presolve must be a boolean")
    for role in config["roles"].values():
        _check_fields("role", role, {"model", "endpoint_env", "key_env"})
    for role in [teacher, *config["roles"].values()]:
        if not isinstance(role["model"], str) or not role["model"]:
            raise ValueError("Specify exact served model names")
        for field in ("endpoint_env", "key_env"):
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", role[field]):
                raise ValueError(
                    "Endpoint/key fields must be environment variable NAMES"
                )
    if not isinstance(teacher["checkpoint"], str) or not teacher["checkpoint"]:
        raise ValueError("An explicit checkpoint identity is required")
    if not isinstance(teacher["tokenizer"], str) or not teacher["tokenizer"]:
        raise ValueError("teacher.tokenizer must name the teacher's tokenizer")
    if teacher["adapter_env"] is not None and not re.fullmatch(
        r"[A-Z_][A-Z0-9_]*", teacher["adapter_env"]
    ):
        raise ValueError("adapter_env must be an environment variable name or null")
    if (
        teacher["format"] not in {"thinking", "non_thinking"}
        or type(teacher["enable_thinking"]) is not bool
    ):
        raise ValueError("Invalid teacher response format / native thinking toggle")
    params = teacher["request_params"]
    if not isinstance(params, dict) or set(params) - {"seed"}:
        raise ValueError(
            "Only seed is supported in teacher.request_params; generation settings belong in protocol"
        )
    if "seed" in params and type(params["seed"]) is not int:
        raise ValueError("Teacher seed must be an integer")
    if provider is not None:
        validate_provider(config)
    evaluation = config["evaluation"]
    for name in ("preferences", "id_preferences"):
        values = evaluation[name]
        if (
            not isinstance(values, list)
            or len(values) != len(set(values))
            or set(values) - set(PREFERENCES)
        ):
            raise ValueError(f"Invalid {name}")
    if not evaluation["preferences"]:
        raise ValueError("Select at least one preference")
    for name in ("attempts", "expected_questions"):
        if type(evaluation[name]) is not int or evaluation[name] < 1:
            raise ValueError(f"Invalid {name}")
    execution = config["execution"]
    for name in (
        "concurrency",
        "student_concurrency",
        "judge_concurrency",
        "episode_error_retries",
    ):
        minimum = 0 if name == "episode_error_retries" else 1
        if type(execution[name]) is not int or execution[name] < minimum:
            raise ValueError(f"Invalid {name}")
    if execution["proxy"] not in {"environment", "direct"}:
        raise ValueError("proxy must be environment or direct")
    if (
        not math.isfinite(execution["episode_timeout_seconds"])
        or execution["episode_timeout_seconds"] <= 0
    ):
        raise ValueError("Invalid episode timeout")


def validate_provider(config: dict) -> None:
    teacher = config["teacher"]
    provider = teacher["provider"]
    _check_fields(
        "teacher.provider",
        provider,
        {"name", "reasoning_effort", "sampling", "output_limit"},
    )
    if provider["name"] not in {"openai", "gemini", "generic"}:
        raise ValueError("teacher.provider.name must be openai, gemini or generic")
    if provider["reasoning_effort"] not in REASONING_EFFORTS:
        raise ValueError("Invalid reasoning effort")
    if provider["sampling"] not in {"config", "provider-default"} or provider[
        "output_limit"
    ] not in {"config", "provider-default"}:
        raise ValueError("Invalid teacher sampling or output-limit policy")
    # API teachers are not seeded and carry no adapter.
    if teacher["request_params"] or teacher["adapter_env"] is not None:
        raise ValueError("A provider teacher takes no request_params or adapter")


def digest(value: str | bytes) -> str:
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def endpoint(role: dict, env: dict) -> str:
    value = env.get(role["endpoint_env"], "").rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"Set a credential-free API base URL in {role['endpoint_env']}"
        )
    return value


def prepare(
    config: dict, args: argparse.Namespace, env: dict
) -> tuple[list[str], dict, dict]:
    limit = args.limit
    shard_count = getattr(args, "shard_count", 1)
    shard_index = getattr(args, "shard_index", 0)
    if limit < 0 or shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid limit or shard selection")
    protocol = yaml.safe_load(Path(config["protocol"]).read_text())
    if not isinstance(protocol, dict) or "defaults" in protocol:
        raise ValueError(
            "Protocol must be a standalone snapshot, not Hydra inheritance"
        )
    teacher, roles, execution = config["teacher"], config["roles"], config["execution"]
    provider = teacher.get("provider")
    concurrency = getattr(args, "concurrency", None) or execution["concurrency"]
    if "presolve" in teacher:
        protocol["evaluator"]["teacher_pre_enabled"] = teacher["presolve"]
    # One protocol student per preference; keep the selected ones, in the order
    # the experiment lists them, on the configured student model.
    students = {
        str(student.get("preference") or "none"): student
        for student in protocol["student_models"]
    }
    missing = [p for p in config["evaluation"]["preferences"] if p not in students]
    if missing:
        raise ValueError(f"Protocol has no student for preferences {missing}")
    protocol["student_models"] = [
        {
            **students[preference],
            "model": roles["student"]["model"],
            "max_concurrent_calls": execution["student_concurrency"],
            "base_url": "${oc.env:EVAL_STUDENT_URL}",
            "api_key": "${oc.env:EVAL_STUDENT_KEY}",
        }
        for preference in config["evaluation"]["preferences"]
    ]
    protocol["auxiliary_model"].update(
        model=roles["judge"]["model"],
        max_concurrent_calls=execution["judge_concurrency"],
    )
    protocol["auxiliary_model"]["base_url"] = "${oc.env:EVAL_AUX_URL}"
    protocol["auxiliary_model"]["api_key"] = "${oc.env:EVAL_AUX_KEY}"
    protocol["teacher_response_format"] = teacher["format"]
    protocol["enable_thinking"] = teacher["enable_thinking"]
    adapter = env.get(teacher["adapter_env"], "") if teacher["adapter_env"] else ""
    if teacher["adapter_env"] and not adapter:
        raise ValueError(
            f"Set {teacher['adapter_env']} to the teacher server's adapter selector"
        )
    urls = {
        name: endpoint(role, env)
        for name, role in {"teacher": teacher, **roles}.items()
    }
    for role in [teacher, *roles.values()]:
        if not env.get(role["key_env"]):
            raise ValueError(
                f"Set {role['key_env']} (EMPTY for an unauthenticated endpoint)"
            )
    env.update(
        EVAL_STUDENT_URL=urls["student"],
        EVAL_STUDENT_KEY=env[roles["student"]["key_env"]],
        EVAL_AUX_URL=urls["judge"],
        EVAL_AUX_KEY=env[roles["judge"]["key_env"]],
        # Context accounting uses the teacher's own tokenizer.
        SHERPA_TOKENIZER=teacher["tokenizer"],
        PYTHONUNBUFFERED="1",
    )
    if execution["proxy"] == "direct":
        for name in PROXY_VARIABLES:
            env.pop(name, None)
    run = {
        "limit": limit,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "concurrency": concurrency,
        "protocol_path": args.output_dir / "protocol.yaml",
        "evaluation_dir": args.output_dir / "evaluation",
        "resume": (args.output_dir / "evaluation/run_config.json").exists(),
    }
    if provider is None:
        command = self_hosted_command(config, protocol, urls, adapter, run, env)
    else:
        command = provider_command(config, args, urls, run, env)
    scientific = copy.deepcopy(config)
    scientific["protocol"] = Path(config["protocol"]).name
    # Operational changes are logged but cannot silently change protocol contents.
    manifest = {
        "schema_version": 2,
        "experiment": scientific,
        "protocol_sha256": digest(yaml.safe_dump(protocol, sort_keys=True)),
        "endpoint_sha256": {name: digest(url) for name, url in urls.items()},
        "adapter_selector_sha256": digest(adapter),
        "asset_location_sha256": {
            "SHERPA_DATASET": digest(env.get("SHERPA_DATASET", "<protocol-default>"))
        },
        "limit": limit,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "source_sha256": {},
    }
    sources = list((REPO / "examples/sherpa").rglob("*.py")) + list(
        (REPO / "examples/common").glob("*.py")
    )
    for field in ("prompts_path", "complaints_path"):
        sources.append(REPO / protocol[f"adaptive_gate_{field}"])
    manifest["source_sha256"] = {
        str(p.relative_to(REPO)): digest(p.read_bytes()) for p in sorted(sources)
    }
    if adapter and Path(adapter).is_dir():
        manifest["adapter_files_sha256"] = {
            p.name: digest(p.read_bytes())
            for p in sorted(Path(adapter).iterdir())
            if p.is_file() and p.suffix in {".json", ".safetensors"}
        }
    return command, manifest_identity(manifest), protocol


def self_hosted_command(
    config: dict, protocol: dict, urls: dict, adapter: str, run: dict, env: dict
) -> list[str]:
    """A teacher you serve: seeded requests, native thinking set explicitly."""
    teacher, execution = config["teacher"], config["execution"]
    params = merge(
        protocol.get("teacher_api_request_params", {}), teacher["request_params"]
    )
    params["extra_body"] = merge(
        params.get("extra_body", {}),
        {"chat_template_kwargs": {"enable_thinking": teacher["enable_thinking"]}},
    )
    # Sampling parameters in a standalone protocol must reach the actual API,
    # while adapter selectors remain deployment-only.
    protocol["teacher_api_request_params"] = copy.deepcopy(params)
    if adapter:
        params["extra_body"]["lora_path"] = adapter
    env["TEACHER_API_KEY"] = env[teacher["key_env"]]
    command = [
        sys.executable,
        "-m",
        "examples.sherpa.eval.evaluator",
        "--config",
        str(run["protocol_path"]),
        "--teacher-base-url",
        urls["teacher"],
        "--teacher-model",
        teacher["model"],
        "--api-key",
        "EMPTY",
        "--teacher-request-params",
        json.dumps(params),
        "--teacher-presolve",
        "config",
        "--attempts",
        str(config["evaluation"]["attempts"]),
        "--limit",
        str(run["limit"]),
        "--shard-count",
        str(run["shard_count"]),
        "--shard-index",
        str(run["shard_index"]),
        "--concurrency",
        str(run["concurrency"]),
        "--episode-timeout-seconds",
        str(execution["episode_timeout_seconds"]),
        "--episode-error-retries",
        str(execution["episode_error_retries"]),
        "--retry-diagnostic-failures",
        "--save-traces",
        "all",
        "--save-api-requests",
        "--output-dir",
        str(run["evaluation_dir"]),
    ]
    if execution["proxy"] == "environment":
        command.append("--keep-env-proxy")
    if run["resume"]:
        command.append("--resume")
    return command


def provider_command(
    config: dict, args: argparse.Namespace, urls: dict, run: dict, env: dict
) -> list[str]:
    """A commercial API teacher: provider request conventions."""
    teacher, roles, execution = config["teacher"], config["roles"], config["execution"]
    provider = teacher["provider"]
    if execution["proxy"] == "environment":
        if env.get("HTTPS_PROXY") or env.get("https_proxy"):
            env.pop("ALL_PROXY", None)
            env.pop("all_proxy", None)
        # The teacher may need the proxy; the self-served student and judge do not.
        bypass = [
            env.get("NO_PROXY", ""),
            env.get("no_proxy", ""),
            "localhost",
            "127.0.0.1",
            "::1",
            urlsplit(urls["student"]).hostname,
            urlsplit(urls["judge"]).hostname,
        ]
        env["NO_PROXY"] = env["no_proxy"] = ",".join(x for x in bypass if x)
    command = [
        sys.executable,
        "-m",
        "examples.sherpa.eval.provider",
        "--config",
        str(run["protocol_path"]),
        "--env-file",
        str(getattr(args, "env_file", REPO / ".env")),
        "--provider",
        provider["name"],
        "--teacher-model",
        teacher["model"],
        "--teacher-base-url",
        urls["teacher"],
        "--teacher-api-key-env",
        teacher["key_env"],
        "--student-base-url",
        urls["student"],
        "--aux-base-url",
        urls["judge"],
        "--student-api-key-env",
        roles["student"]["key_env"],
        "--aux-api-key-env",
        roles["judge"]["key_env"],
        "--reasoning-effort",
        provider["reasoning_effort"],
        "--teacher-format",
        teacher["format"],
        "--teacher-sampling",
        provider["sampling"],
        "--teacher-output-limit",
        provider["output_limit"],
        "--concurrency",
        str(run["concurrency"]),
        "--episode-timeout-seconds",
        str(execution["episode_timeout_seconds"]),
        "--output-dir",
        str(run["evaluation_dir"]),
    ]
    if run["resume"]:
        command.append("--resume")
    command += [
        "--",
        "--limit",
        str(run["limit"]),
        "--attempts",
        str(config["evaluation"]["attempts"]),
        "--shard-count",
        str(run["shard_count"]),
        "--shard-index",
        str(run["shard_index"]),
        "--teacher-presolve",
        "config",
        "--save-traces",
        "all",
        "--skip-preflight",
        "--save-api-requests",
        "--episode-error-retries",
        str(execution["episode_error_retries"]),
        "--retry-diagnostic-failures",
    ]
    if execution["proxy"] == "environment":
        command.append("--keep-env-proxy")
    return command


def manifest_identity(manifest: dict) -> dict:
    """Exclude this orchestration-only runner.

    The generated protocol, request/deployment settings, and actual evaluator
    source remain strictly checked.
    """
    result = copy.deepcopy(manifest)
    result["source_sha256"] = {
        path: value
        for path, value in result.get("source_sha256", {}).items()
        if path != "examples/sherpa/eval/runner.py"
    }
    return result


def check_manifest(path: Path, expected: dict) -> None:
    if path.exists() and resume_identity(
        json.loads(path.read_text())
    ) != resume_identity(expected):
        raise ValueError(
            "Experiment/protocol/deployment changed; use a new output directory"
        )


def resume_identity(manifest: dict) -> dict:
    """Episode concurrency is operational, not identity.

    Keep actual values in manifests and invocation history for reproducibility.
    Per-caller limits, timeout/retry policy, and all scientific fields stay strict.
    """
    result = manifest_identity(manifest)
    execution = result.get("experiment", {}).get("execution", {})
    execution.pop("concurrency", None)
    return result


def preflight(config: dict, env: dict) -> None:
    import httpx

    roles = dict(config["roles"])
    if config["teacher"].get("provider") is None:
        # Commercial catalogs need not list the model id; their calls fail loudly.
        roles = {"teacher": config["teacher"], **roles}
    with httpx.Client(
        trust_env=config["execution"]["proxy"] == "environment", timeout=30
    ) as client:
        for name, role in roles.items():
            response = client.get(
                endpoint(role, env) + "/models",
                headers={"Authorization": f"Bearer {env[role['key_env']]}"},
            )
            if response.status_code != 200 or role["model"] not in {
                v["id"] for v in response.json().get("data", [])
            }:
                raise RuntimeError(
                    f"{name} model catalog preflight failed (HTTP {response.status_code})"
                )


def completion_status(output: Path, config: dict, args: argparse.Namespace) -> dict:
    questions = (
        min(args.limit, config["evaluation"]["expected_questions"])
        if args.limit
        else config["evaluation"]["expected_questions"]
    )
    rows = questions * len(config["evaluation"]["preferences"])
    expected = (
        len(range(args.shard_index, rows, args.shard_count))
        * config["evaluation"]["attempts"]
    )
    path = output / "evaluation/summary.json"
    if not path.exists():
        return {"complete": False, "expected": expected, "reason": "missing summary"}
    summary = json.loads(path.read_text())
    protocol = yaml.safe_load(Path(config["protocol"]).read_text())
    enabled = config["teacher"].get(
        "presolve", protocol["evaluator"].get("teacher_pre_enabled")
    )
    if enabled is None:
        enabled = protocol["teacher_pre"]["enabled"]
    mode = summary.get("modes", {}).get(
        "presolve_on" if enabled else "presolve_off", {}
    )
    completed = mode.get("completed_attempts", 0)
    pending = summary.get("pending_backfill", {}).get("count", 0)
    return {
        "complete": completed == expected and pending == 0,
        "expected": expected,
        "completed": completed,
        "pending": pending,
    }


def main() -> None:
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Config name in configs/ or a YAML path")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Problems before the per-preference expansion; 0 = all",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--concurrency", type=int, help="Episodes in flight (default: config)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="No API calls or output writes"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check model catalogs; no generation or output writes",
    )
    args = parser.parse_args()
    try:
        path = Path(args.config)
        if not path.is_file():
            path = PACKAGE / "configs" / (args.config.removesuffix(".yaml") + ".yaml")
        config = load_config(path)
        if args.concurrency is not None and args.concurrency < 1:
            raise ValueError("Concurrency must be positive")
        args.env_file = args.env_file.resolve()
        load_dotenv(args.env_file, override=False)
        args.output_dir = (
            args.output_dir or REPO / "output/eval" / config["run_name"]
        ).resolve()
        env = dict(os.environ)
        command, manifest, protocol = prepare(config, args, env)
        if args.dry_run or args.preflight:
            if args.preflight:
                preflight(config, env)
            sys.stdout.write(json.dumps(manifest, indent=2) + "\n")
            return
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        lock_path = args.output_dir.with_name(args.output_dir.name + ".lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            marker = args.output_dir / "experiment.json"
            check_manifest(marker, manifest)
            if (
                args.output_dir.exists()
                and any(args.output_dir.iterdir())
                and not marker.exists()
            ):
                raise ValueError(
                    "Refusing to adopt an existing output without a matching manifest"
                )
            preflight(config, env)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            protocol_path = args.output_dir / "protocol.yaml"
            text = yaml.safe_dump(protocol, sort_keys=False)
            if protocol_path.exists() and protocol_path.read_text() != text:
                raise ValueError("Generated protocol was modified; refusing resume")
            if not protocol_path.exists():
                protocol_path.write_text(text)
            marker.write_text(json.dumps(manifest, indent=2) + "\n")
            with (args.output_dir / "invocations.jsonl").open("a") as events:
                events.write(
                    json.dumps(
                        {
                            "at": datetime.now(UTC).isoformat(),
                            "resume": "--resume" in command,
                            "execution": config["execution"],
                            "concurrency": args.concurrency,
                        }
                    )
                    + "\n"
                )
            with (args.output_dir / "console.log").open("ab") as log:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                try:
                    for line in process.stdout:
                        sys.stdout.buffer.write(line)
                        sys.stdout.buffer.flush()
                        log.write(line)
                        log.flush()
                    status = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if status:
                raise SystemExit(status)
            completion = completion_status(args.output_dir, config, args)
            (args.output_dir / "completion.json").write_text(
                json.dumps(completion, indent=2)
            )
            if not completion["complete"]:
                raise RuntimeError(
                    "Evaluation has missing/pending episodes; repeat the same command to resume"
                )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
