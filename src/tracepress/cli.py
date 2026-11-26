"""The `tracepress` command: `tracepress serve --port 7700`."""

from __future__ import annotations

import argparse
import logging
import os


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tracepress")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run a server that keeps registered pipelines running on a live stream")
    serve.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: localhost only)")
    serve.add_argument("--port", type=int, default=7700)
    serve.add_argument(
        "--token",
        default=os.environ.get("TRACEPRESS_TOKEN") or None,
        help="require this token from clients; needed for any non-local --host (default: $TRACEPRESS_TOKEN)",
    )
    serve.add_argument(
        "--retain",
        type=int,
        default=None,
        help="events to keep per table, replayed to pipelines registered later (default: all)",
    )
    serve.add_argument(
        "--lm", help='default LM for pipelines that don\'t name one, e.g. "openrouter:openai/gpt-6-luna"'
    )
    args = parser.parse_args(argv)

    if args.command == "serve":
        from .server.app import LOCAL_HOSTS
        from .server.app import serve as run

        if args.host not in LOCAL_HOSTS and not args.token:
            parser.error(
                f"--host {args.host} lets anyone who can reach it run code here; set TRACEPRESS_TOKEN (or pass --token) too"
            )
        if args.lm:
            from .lm import configure

            configure(lm=args.lm)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        print(
            f"tracepress server on {args.host}:{args.port}. Registered pipelines run pickled code; "
            "only connect clients you trust.",
            flush=True,
        )
        run(args.host, args.port, args.token, args.retain)


if __name__ == "__main__":
    main()
