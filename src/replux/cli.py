"""Validate startup configuration before creating the chat interface."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import httpx2 as httpx

from replux.config import ConfigError, load_config
from replux.llm import create_client
from replux.tools.process import BashNotFound, require_bash
from replux.tui import RepluxApp


def create_app(
    argv: Sequence[str] | None = None, *, transport: httpx.AsyncBaseTransport | None = None
) -> RepluxApp:
    parser = argparse.ArgumentParser(prog="replux", description="Replux terminal AI assistant")
    parser.add_argument("--config", type=Path, help="Configuration file (default: ~/.replux/config.yaml)")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        bash = require_bash()
    except (ConfigError, BashNotFound) as error:
        parser.error(str(error))
    return RepluxApp(create_client(config, transport=transport, bash=bash), model=config.model)


def main() -> None:
    app = create_app()
    app.run()
    raise SystemExit(app.return_code)
