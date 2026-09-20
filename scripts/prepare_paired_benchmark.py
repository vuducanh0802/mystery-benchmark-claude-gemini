#!/usr/bin/env python3
"""Restore and verify the frozen Claude/Gemini suite without regenerating cases."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def prepare(target):
    bundle = ROOT / "benchmark_suites/claude_gemini_cases_v1.tar.gz"
    metadata = json.loads(bundle.with_name("claude_gemini_cases_v1.json").read_text())
    raw = bundle.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["archive_sha256"]:
        raise ValueError("Bundle checksum mismatch")
    # Validate every file and any existing destination before writing anything.
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unexpected archive member")
            if member.name in payloads or member.name not in metadata["files"]:
                raise ValueError("Unexpected or duplicate archive member")
            data = archive.extractfile(member).read()
            expected = metadata["files"][member.name]
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Case checksum mismatch: {member.name}")
            destination = target / relative
            if target.resolve() not in destination.resolve().parents:
                raise ValueError("Destination escapes benchmark directory")
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Existing suite differs at {relative}; choose a new BENCHMARK_DIR")
            payloads[member.name] = data
    if set(payloads) != set(metadata["files"]):
        raise ValueError("Incomplete bundle")
    for relative, data in payloads.items():
        destination = target / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    print(f"Verified {metadata['case_entries']} frozen case entries at {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmark_v1")
    args = parser.parse_args()
    prepare(args.output_dir)
