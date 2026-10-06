"""Resolve the pinned official BF16 checkpoint from the local HF cache; never download."""

from huggingface_hub import snapshot_download

MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
REVISION = "f6fae9795746f63c9be8344932f01275f3c63734"


def resolve_local_model() -> str:
    try:
        return snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"{MODEL_ID} at revision {REVISION} is not in the local Hugging Face "
            f"cache. Download it first (hf download {MODEL_ID} --revision "
            f"{REVISION}) or set MINISTRAL_MODEL_PATH to a full local snapshot. "
            "No download was attempted."
        ) from exc


if __name__ == "__main__":
    print(resolve_local_model())
