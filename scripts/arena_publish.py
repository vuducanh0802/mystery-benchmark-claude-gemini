"""Package or publish a MysteryArena run to a Hugging Face Dataset repo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.hf_publish import HFPublishError, package_run_for_hf, publish_run_to_hf


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a MysteryArena run to Hugging Face")
    parser.add_argument("run_dir", help="Local Arena run directory, e.g. arena/results/run_001")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("ARENA_HF_DATASET"),
        help="Hugging Face dataset repo id. Defaults to ARENA_HF_DATASET.",
    )
    parser.add_argument(
        "--package-dir",
        default=None,
        help="Optional local staging directory. Existing contents are replaced.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Only build the Hugging Face dataset folder locally.",
    )
    parser.add_argument("--private", action="store_true", help="Create the dataset as private.")
    parser.add_argument("--revision", default=None, help="Target branch or revision.")
    parser.add_argument("--create-pr", action="store_true", help="Upload as a pull request.")
    parser.add_argument(
        "--no-model-responses",
        action="store_true",
        help="Strip raw model_response fields from published trajectories.",
    )
    args = parser.parse_args()

    try:
        if args.no_upload:
            package = package_run_for_hf(
                args.run_dir,
                package_dir=args.package_dir,
                include_model_responses=not args.no_model_responses,
            )
            payload = package.to_dict()
        else:
            payload = publish_run_to_hf(
                args.run_dir,
                repo_id=args.repo_id,
                private=args.private,
                revision=args.revision,
                create_pr=args.create_pr,
                package_dir=args.package_dir,
                include_model_responses=not args.no_model_responses,
            )
    except HFPublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
