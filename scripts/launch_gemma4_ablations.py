#!/usr/bin/env python3
"""Serve cached Gemma replicas, verify a pilot, then run/resume the full ablations."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import run_gemma4_ablations as gemma
from scripts import run_claude_gemini_baselines as core
from scripts.prepare_paired_benchmark import prepare

GGUF_SHA256 = "38bd64c852c4b460434cc7162fa9bdcf242faf86502581a754cb72956bb17f84"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama-server", type=Path, required=True)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--expected-gguf-sha256", default=GGUF_SHA256)
    p.add_argument("--gpus", nargs="+", required=True, help="Unused physical GPU indices, one replica each")
    p.add_argument("--ports", nargs="+", type=int, default=[8332, 8333, 8334])
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--context", type=int, default=8192, help="Context tokens per server slot")
    p.add_argument("--workers", type=int, help="Default: parallel slots times replica count")
    p.add_argument("--benchmark-dir", type=Path, default=ROOT / "data/benchmark_v1")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--experiment-id", default="gemma4_guarded_ablations_v1")
    p.add_argument("--pilot-per-level", type=int, default=1)
    p.add_argument("--include-guarded-reference", action="store_true")
    p.add_argument("--server-start-timeout", type=float, default=600)
    args = p.parse_args(argv)
    args.gpus = [str(int(gpu)) for gpu in args.gpus]
    if (len(args.gpus) != len(args.ports) or len(set(args.gpus)) != len(args.gpus)
            or len(set(args.ports)) != len(args.ports)):
        p.error("Choose distinct GPU indices and one distinct port per GPU")
    if min(args.parallel, args.context) <= 0 or args.pilot_per_level < 0:
        p.error("Invalid parallel, context or pilot size")
    if any(not 1024 <= port <= 65535 for port in args.ports):
        p.error("Ports must be between 1024 and 65535")
    if args.server_start_timeout <= 0:
        p.error("Server start timeout must be positive")
    args.workers = args.workers or args.parallel * len(args.gpus)
    if args.workers <= 0:
        p.error("workers must be positive")
    for name in ("llama_server", "gguf", "benchmark_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def server_command(args, port):
    return [
        str(args.llama_server), "--model", str(args.gguf), "--alias", gemma.MODEL,
        "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(args.context * args.parallel), "--parallel", str(args.parallel),
        "--cont-batching", "--flash-attn", "on", "--n-gpu-layers", "999",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--batch-size", "4096", "--ubatch-size", "1024",
        "--cache-reuse", "256", "--cache-ram", "4096",
        "--threads", "8", "--threads-batch", "8", "--threads-http", "24",
        "--jinja", "--reasoning", "off", "--no-context-shift", "--no-webui", "--metrics",
    ]


def check_resources(args):
    if not args.llama_server.is_file() or not os.access(args.llama_server, os.X_OK):
        raise ValueError(f"Missing executable: {args.llama_server}")
    if not args.gguf.is_file():
        raise ValueError(f"Missing GGUF: {args.gguf}")
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits",
    ], text=True)
    memory = {index.strip(): int(used.strip()) for index, used in
              (line.split(",") for line in raw.splitlines())}
    for gpu in args.gpus:
        if gpu not in memory or memory[gpu] > 512:
            raise ValueError(f"GPU {gpu} is unavailable or already occupied; existing processes were not modified")
    for port in args.ports:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))


def runner_command(args, output_dir, experiment_id, per_level=200):
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts/run_gemma4_ablations.py"),
        "--benchmark-dir", str(args.benchmark_dir), "--output-dir", str(output_dir),
        "--experiment-id", experiment_id, "--per-level", str(per_level),
        "--workers", str(args.workers), "--base-urls",
        *[f"http://127.0.0.1:{port}/v1" for port in args.ports],
        "--serving-metadata", str(args.output_dir / "serving.json"),
    ]
    if args.include_guarded_reference:
        cmd.append("--include-guarded-reference")
    return cmd


def main(argv=None):
    args = parse_args(argv)
    check_resources(args)
    prepare(args.benchmark_dir)
    print("Checking cached GGUF checksum...", flush=True)
    gguf_hash = core._sha256(args.gguf)
    if gguf_hash != args.expected_gguf_sha256:
        raise ValueError("GGUF checksum differs from the requested checkpoint")
    commands = [server_command(args, port) for port in args.ports]
    known_checkpoint = gguf_hash == GGUF_SHA256
    libraries = sorted({p.resolve() for p in args.llama_server.parent.glob("*.so*") if p.is_file()})
    metadata = {
        "model": gemma.MODEL, "quantization": "GGUF Q4_K_M" if known_checkpoint else "unverified",
        "source_revision": gemma.SOURCE_REVISION if known_checkpoint else "unspecified",
        "inference_revision": gemma.INFERENCE_REVISION if known_checkpoint else "unspecified",
        "gguf_sha256": gguf_hash, "gguf_path": str(args.gguf),
        "llama_server": str(args.llama_server),
        "llama_server_sha256": core._sha256(args.llama_server),
        "shared_library_sha256": {p.name: core._sha256(p) for p in libraries},
        "llama_server_version": subprocess.check_output(
            [str(args.llama_server), "--version"], stderr=subprocess.STDOUT, text=True, timeout=30).strip(),
        "reasoning": "off", "context_per_slot": args.context,
        "parallel_slots_per_server": args.parallel, "gpus": args.gpus,
        "server_commands": commands,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "serving.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Existing serving provenance differs; use a new output directory")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    # Verify the complete matrix before loading weights onto GPUs.
    subprocess.run(runner_command(args, args.output_dir, args.experiment_id) + ["--validate-only"], check=True)
    log_dir = args.output_dir / "server_logs"
    log_dir.mkdir(exist_ok=True)
    owned = []
    status_path = args.output_dir / "launcher_status.json"
    def status(stage, **extra):
        status_path.write_text(json.dumps({"stage": stage, "updated_at": core._now(), **extra}, indent=2))
        print(f"[{core._now()}] {stage}", flush=True)
    try:
        status("starting_servers")
        for gpu, port, command in zip(args.gpus, args.ports, commands):
            handle = (log_dir / f"gpu_{gpu}_port_{port}.log").open("a")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            proc = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            owned.append((proc, handle, gpu, port))
        deadline = time.monotonic() + args.server_start_timeout
        ready = set()
        while len(ready) != len(owned):
            for proc, handle, gpu, port in owned:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server GPU {gpu} exited ({proc.returncode}); see {handle.name}")
                if port in ready:
                    continue
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                        payload = json.loads(response.read())
                    if gemma.MODEL in {item.get("id") for item in payload.get("data", [])}:
                        ready.add(port)
                        print(f"GPU {gpu} ready on port {port}", flush=True)
                except (OSError, ValueError):
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError("Model servers did not become ready in time")
            if len(ready) != len(owned):
                time.sleep(2)
        status("preflight")
        subprocess.run(runner_command(args, args.output_dir, args.experiment_id) + ["--preflight-only"], check=True)
        if args.pilot_per_level:
            status("pilot")
            subprocess.run(runner_command(
                args, args.output_dir / "pilot", args.experiment_id + "_pilot", args.pilot_per_level), check=True)
        status("full_run", expected_episodes=1000 * (4 if args.include_guarded_reference else 3))
        completed = subprocess.run(runner_command(args, args.output_dir, args.experiment_id))
        status("complete" if completed.returncode == 0 else "incomplete", exit_code=completed.returncode)
        return completed.returncode
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Only terminate subprocesses created by this launcher.
        for proc, _, _, _ in owned:
            if proc.poll() is None:
                proc.terminate()
        for proc, handle, _, _ in owned:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            handle.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
