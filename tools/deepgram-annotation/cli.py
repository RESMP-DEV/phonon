#!/usr/bin/env python3
"""Command-line interface for the offline Deepgram annotation builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deepgram_client import ARMS, DEFAULT_BASE_URL
from manifest import build_manifest, load_records, write_jsonl
from runner import RunnerError, run


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the annotation CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Build group-disjoint manifests and run Deepgram annotation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Build the group-disjoint train/eval manifest."
    )
    build_parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Source JSONL manifest(s).",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL manifest path.",
    )
    build_parser.add_argument(
        "--report",
        type=Path,
        help="Optional aggregate report JSON path.",
    )
    build_parser.add_argument(
        "--train-target-ms",
        type=int,
        help="Exact train duration target in milliseconds.",
    )
    build_parser.add_argument(
        "--eval-target-ms",
        type=int,
        help="Exact eval duration target in milliseconds.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run the four-arm Deepgram annotation for train/eval rows.",
    )
    run_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Group-disjoint manifest JSONL produced by the build verb.",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Append-only annotation output JSONL.",
    )
    run_parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Directory holding the raw provider responses.",
    )
    run_parser.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="Aggregate-only receipt JSON path.",
    )
    run_parser.add_argument(
        "--arms",
        nargs="+",
        choices=sorted(ARMS),
        default=sorted(ARMS),
        help="Arms to annotate.",
    )
    run_parser.add_argument(
        "--keyterms",
        type=Path,
        help="Keyterm profile JSON used by the keyterm arms.",
    )
    run_parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Deepgram base URL (default: {DEFAULT_BASE_URL}).",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        choices=range(1, 6),
        help="Concurrent requests; bounded to 5 by the provider.",
    )
    run_parser.add_argument(
        "--keyring-profile",
        default="default",
        help="deepctl keyring profile holding the API key.",
    )
    return parser.parse_args(argv)


def build_command(arguments: argparse.Namespace) -> int:
    """Build a group-disjoint manifest from source records."""
    records = load_records(arguments.input)
    train_target_ms = arguments.train_target_ms
    eval_target_ms = arguments.eval_target_ms
    if train_target_ms is None or eval_target_ms is None:
        total_ms = sum(
            row["duration_milliseconds"]
            for row in records
            if row.get("declared_split") != "test"
        )
        if train_target_ms is None and eval_target_ms is None:
            train_target_ms = round(total_ms * 0.8)
            eval_target_ms = total_ms - train_target_ms
        elif train_target_ms is None:
            train_target_ms = total_ms - eval_target_ms
        else:
            eval_target_ms = total_ms - train_target_ms
    manifest, report = build_manifest(
        records,
        train_target_ms=train_target_ms,
        eval_target_ms=eval_target_ms,
    )
    write_jsonl(arguments.output, manifest)
    if arguments.report is not None:
        import json

        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(
        f"built {report['rows']} rows into {arguments.output} "
        f"(train {report['train_rows']}, eval {report['eval_rows']})"
    )
    return 0


def run_command(arguments: argparse.Namespace) -> int:
    """Run the annotation plan over an existing group-disjoint manifest."""
    try:
        result = run(
            arguments.manifest,
            arguments.output,
            arguments.archive,
            arguments.receipt,
            arms=arguments.arms,
            keyterm_path=arguments.keyterms,
            base_url=arguments.base_url,
            concurrency=arguments.concurrency,
        )
    except RunnerError as error:
        print(f"annotation run failed: {error}", file=sys.stderr)
        return 1
    print(
        f"annotation run finished: completed={result.completed} "
        f"failed={result.failed} skipped={result.skipped}"
    )
    return 0 if result.failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    arguments = parse_arguments(argv)
    if arguments.command == "build":
        return build_command(arguments)
    if arguments.command == "run":
        return run_command(arguments)
    raise AssertionError(f"unhandled command {arguments.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
