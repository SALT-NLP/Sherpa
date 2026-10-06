#!/usr/bin/env python3
"""Fetch the small, pinned subset of upstream MathTutorBench needed at runtime."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UPSTREAM_COMMIT = "6faed173ec2bef55cb899b2a3e0f93982f9cb176"
RAW_ROOT = f"https://raw.githubusercontent.com/eth-lre/mathtutorbench/{UPSTREAM_COMMIT}"

# Exact Git blob sizes at UPSTREAM_COMMIT; a size mismatch marks a partial download.
UPSTREAM_FILES = {
    "README.md": 14_935,
    "requirements.txt": 190,
    "registry.py": 494,
    "configs/mistake_correction.yaml": 611,
    "configs/mistake_location.yaml": 1_433,
    "configs/pedagogy_following.yaml": 771,
    "configs/pedagogy_following_hard.yaml": 781,
    "configs/problem_solving.yaml": 1_003,
    "configs/scaffolding_generation.yaml": 491,
    "configs/scaffolding_generation_hard.yaml": 501,
    "configs/socratic_questioning.yaml": 681,
    "configs/student_solution_correctness.yaml": 1_021,
    "dataloaders/__init__.py": 0,
    "dataloaders/base.py": 1_021,
    "dataloaders/mathbridge.py": 1_626,
    "datasets/mathdial_bridge.json": 1_550_455,
    "datasets/mathdial_bridge_hard.json": 718_256,
    "tasks/__init__.py": 478,
    "tasks/base.py": 2_055,
    "tasks/extraction.py": 908,
    "tasks/gsm8k.py": 3_744,
    "tasks/mistake_correction.py": 3_895,
    "tasks/mistake_location.py": 4_154,
    "tasks/pedagogy_following.py": 1_258,
    "tasks/pedagogy_following_hard.py": 1_267,
    "tasks/scaffolding_generation.py": 1_265,
    "tasks/scaffolding_generation_hard.py": 1_274,
    "tasks/socratic_questioning.py": 2_205,
    "tasks/solution_correctness.py": 4_252,
}


def valid_file(root: Path, relative: str, size: int) -> bool:
    path = root / relative
    return path.is_file() and path.stat().st_size == size


def download_one(root: Path, relative: str, size: int) -> str:
    destination = root / relative
    if valid_file(root, relative, size):
        return relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    request = urllib.request.Request(
        f"{RAW_ROOT}/{relative}",
        headers={"User-Agent": "AReaL-MathTutorBench-runner"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if len(payload) != size:
            raise RuntimeError(
                f"unexpected size for {relative}: got {len(payload)}, expected {size}"
            )
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return relative


def prepare_upstream(root: Path, offline: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    missing = [
        (relative, size)
        for relative, size in UPSTREAM_FILES.items()
        if not valid_file(root, relative, size)
    ]
    if missing:
        if offline:
            raise SystemExit(
                "offline preparation is missing pinned MathTutorBench files: "
                + ", ".join(relative for relative, _ in missing)
            )
        print(
            f"[prepare] downloading {len(missing)} pinned upstream files "
            f"({sum(size for _, size in missing) / 1_000_000:.2f} MB)"
        )
        with ThreadPoolExecutor(max_workers=min(8, len(missing))) as pool:
            futures = {
                pool.submit(download_one, root, relative, size): relative
                for relative, size in missing
            }
            for future in as_completed(futures):
                future.result()
    revision_file = root / ".math_tutor_bench_revision"
    revision_file.write_text(UPSTREAM_COMMIT + "\n", encoding="utf-8")
    invalid = [
        relative
        for relative, size in UPSTREAM_FILES.items()
        if not valid_file(root, relative, size)
    ]
    if invalid:
        raise SystemExit(f"upstream preparation is incomplete: {invalid}")
    print(f"[prepare] upstream ready at {root} ({UPSTREAM_COMMIT[:8]})")


def dependency_imports_work(python: Path, dependency_root: Path) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(dependency_root), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [str(python), "-c", "import sacrebleu, portalocker"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def prepare_dependencies(python: Path, dependency_root: Path, offline: bool) -> None:
    dependency_root.mkdir(parents=True, exist_ok=True)
    if dependency_imports_work(python, dependency_root):
        print(f"[prepare] Python dependencies ready at {dependency_root}")
        return
    if offline:
        raise SystemExit(
            "offline preparation is missing private dependencies: "
            "sacrebleu and/or portalocker"
        )
    # uv, not pip: a uv-created venv ships without pip.
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required to install sacrebleu and portalocker")
    print("[prepare] installing sacrebleu and portalocker into the private runtime")
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--target",
            str(dependency_root),
            "sacrebleu==2.5.1",
            "portalocker>=2.7",
        ],
        check=True,
    )
    if not dependency_imports_work(python, dependency_root):
        raise SystemExit("private MathTutorBench dependencies are still not importable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if not args.python.is_file():
        raise SystemExit(f"Python interpreter does not exist: {args.python}")
    prepare_upstream(args.upstream.resolve(), args.offline)
    # Keep the venv's python symlink intact: Path.resolve() would follow it to the
    # base interpreter, which does not see the venv's installed packages.
    python = Path(os.path.abspath(args.python))
    prepare_dependencies(python, args.dependency_root.resolve(), args.offline)


if __name__ == "__main__":
    main()
