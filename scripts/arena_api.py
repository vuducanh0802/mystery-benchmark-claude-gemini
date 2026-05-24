"""Launch the MysteryArena backend API server."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from arena.api import create_app


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pick_port(host: str, requested: int, scan: int, strict: bool) -> int:
    if strict or _port_available(host, requested):
        return requested
    for port in range(requested + 1, requested + scan + 1):
        if _port_available(host, port):
            print(f"Port {requested} is busy; using {port} instead.")
            return port
    raise OSError(
        f"Cannot find empty port in range: {requested}-{requested + scan}. "
        "Pass --port with a free port, or close the existing API server."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MysteryArena backend API")
    parser.add_argument("--arena-root", default="arena/results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--port-scan", type=int, default=50)
    parser.add_argument("--strict-port", action="store_true")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Load gateway/API variables from this dotenv file if present.",
    )
    parser.add_argument("--gateway-url", default=None)
    parser.add_argument("--gateway-key-env", default=None)
    parser.add_argument("--gateway-model", default=None)
    parser.add_argument("--gateway-url-env", default="LLM_GATEWAY_URL")
    parser.add_argument("--gateway-key-env-default", default="LLM_GATEWAY_API_KEY")
    args = parser.parse_args()

    app = create_app(
        arena_root=args.arena_root,
        env_file=args.env_file,
        gateway_url=args.gateway_url,
        gateway_key_env=args.gateway_key_env,
        gateway_model=args.gateway_model,
        gateway_url_env=args.gateway_url_env,
        gateway_key_env_default=args.gateway_key_env_default,
    )
    port = _pick_port(args.host, args.port, args.port_scan, args.strict_port)
    print(f"MysteryArena API: http://{args.host}:{port}")
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
