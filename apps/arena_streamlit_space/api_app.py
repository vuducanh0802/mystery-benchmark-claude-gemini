from __future__ import annotations

import os

from arena.api import create_app


app = create_app(
    arena_root=os.environ.get("ARENA_ROOT", "/data/arena/results"),
    env_file=os.environ.get("ARENA_ENV_FILE") or None,
)
