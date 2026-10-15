"""Validate startup configuration before creating the chat interface."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import httpx2 as httpx

from replux.config import ConfigError, load_config
from replux.llm import create_client
from replux.tui import RepluxApp


def create_app(
    argv: Sequence[str] | None = None, *, transport: httpx.AsyncBaseTransport | None = None
) -> RepluxApp:
    parser = argparse.ArgumentParser(prog="replux", description="Replux 终端 AI 助手")
    parser.add_argument("--config", type=Path, help="配置文件（默认 ~/.replux/config.yaml）")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as error:
        parser.error(str(error))
    return RepluxApp(create_client(config, transport=transport))


def main() -> None:
    app = create_app()
    app.run()
    raise SystemExit(app.return_code)
