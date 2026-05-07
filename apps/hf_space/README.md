---
title: MysteryArena Textual
emoji: "🕵"
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# MysteryArena — Textual World (HF Space)

Interactive murder-mystery game. NPCs are powered by the OpenAI ChatGPT API.

## Deploy

1. Create a new Space (Gradio SDK).
2. Push this directory (`apps/hf_space/`) as the Space root, **or** use a Space
   that points at this subfolder (set `app_file: apps/hf_space/app.py`).
3. Add an `OPENAI_API_KEY` Space secret.
4. Optionally set `MYSTERY_NPC_MODEL` (default: `gpt-4o-mini`).

## Run locally

```
uv run apps/hf_space/app.py
```

The UI exposes level, seed, transcript, command box, and an accusation panel.
Type `help` in the command box for the command list.
