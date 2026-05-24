"""Create or update the MysteryArena Streamlit Hugging Face Space."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("agents", "arena", "evaluation", "mystery_world")
SCRIPT_FILES = (
    "arena_run.py",
    "arena_client.py",
    "run_arena_matches.py",
)


def _load_env_file(path: str | Path | None) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _space_url(space_id: str) -> str:
    return f"https://{space_id.replace('/', '-').replace('_', '-').lower()}.hf.space"


def _ignore_source(_: str, names: list[str]) -> set[str]:
    ignored = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".DS_Store",
    }
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def _prepare_upload_dir(space_dir: Path) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temp = tempfile.TemporaryDirectory(prefix="arena_space_upload_")
    upload_dir = Path(temp.name)
    shutil.copytree(space_dir, upload_dir, dirs_exist_ok=True, ignore=_ignore_source)
    for source_dir in SOURCE_DIRS:
        shutil.copytree(
            PROJECT_ROOT / source_dir,
            upload_dir / source_dir,
            dirs_exist_ok=True,
            ignore=_ignore_source,
        )
    scripts_dir = upload_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for filename in SCRIPT_FILES:
        shutil.copy2(PROJECT_ROOT / "scripts" / filename, scripts_dir / filename)
    return temp, upload_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy the Arena full-stack frontend/backend to HF Spaces")
    parser.add_argument("space_id", help="Target Space id, e.g. org/mystery-arena")
    parser.add_argument(
        "--dataset-repo",
        default=None,
        help="Dataset repo the Space reads. Defaults to ARENA_HF_DATASET from --env-file or environment.",
    )
    parser.add_argument("--env-file", default=".env", help="Dotenv file used for HF_TOKEN and Space secrets.")
    parser.add_argument("--revision", default="main", help="Dataset revision read by the Space.")
    parser.add_argument("--default-run", default="latest", help="Initial run id or 'latest'.")
    parser.add_argument("--private", action="store_true", help="Create the Space as private.")
    parser.add_argument(
        "--space-dir",
        default="apps/arena_streamlit_space",
        help="Local Streamlit Space directory to upload.",
    )
    parser.add_argument("--create-pr", action="store_true", help="Upload as a Space pull request.")
    parser.add_argument(
        "--sync-secrets",
        action="store_true",
        help="Also sync backend/publish secrets from --env-file into the Space. Off by default.",
    )
    args = parser.parse_args()

    _load_env_file(args.env_file)
    args.dataset_repo = args.dataset_repo or os.environ.get("ARENA_HF_DATASET")
    if not args.dataset_repo:
        print("error: --dataset-repo is required or ARENA_HF_DATASET must be set", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: huggingface_hub is required", file=sys.stderr)
        return 2

    space_dir = Path(args.space_dir)
    if not space_dir.exists() or not space_dir.is_dir():
        print(f"error: Space directory not found: {space_dir}", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    for key, value in {
        "ARENA_DATASET_REPO": args.dataset_repo,
        "ARENA_HF_DATASET": args.dataset_repo,
        "ARENA_DEFAULT_REVISION": args.revision,
        "ARENA_API_URL": _space_url(args.space_id),
        "ARENA_API_PUBLIC_URL": _space_url(args.space_id),
        "ARENA_ROOT": "/data/arena/results",
    }.items():
        api.add_space_variable(repo_id=args.space_id, key=key, value=value)

    if args.sync_secrets:
        for key in (
            "HF_TOKEN",
            "LLM_GATEWAY_URL",
            "LLM_GATEWAY_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            value = os.environ.get(key)
            if value:
                api.add_space_secret(repo_id=args.space_id, key=key, value=value)

    temp_upload, upload_dir = _prepare_upload_dir(space_dir)
    try:
        upload = api.upload_folder(
            repo_id=args.space_id,
            repo_type="space",
            folder_path=str(upload_dir),
            ignore_patterns=["__pycache__/*", "*.pyc"],
            create_pr=args.create_pr,
            commit_message="Deploy MysteryArena full-stack Space",
        )
    finally:
        temp_upload.cleanup()
    print(f"Space: https://huggingface.co/spaces/{args.space_id}")
    print(f"Upload: {upload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
