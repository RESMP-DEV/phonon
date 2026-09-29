from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from phonon.eval import evaluate_dataset, export_eval_pack
from phonon.ingest_aqua import ingest_aqua
from phonon.ingest_hf import download_hf_dataset, install_curated_config
from phonon.ingest_youtube import fetch_youtube_audio, ingest_youtube_video
from phonon.label import make_label_queue
from phonon.report import build_report
from phonon.review_ui import build_review_ui
from phonon.teachers import generate_teacher_transcripts, merge_teacher_sidecar_into_dataset
from phonon.vendor import list_openai_compatible_models
from phonon.vocab_resolver import RepoVocabResolver
from phonon.youtube_batch import ingest_youtube_candidates


DEFAULT_DATASET_ROOT = Path("/home/user/phonon/datasets")


def add_common_dataset_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)


def add_subcommand_dataset_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=argparse.SUPPRESS)


def cmd_doctor(args: argparse.Namespace) -> None:
    checks = {
        "dataset_root": str(args.dataset_root),
        "dataset_root_exists": args.dataset_root.exists(),
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "yt-dlp": shutil.which("yt-dlp"),
        "hf": shutil.which("hf") or shutil.which("huggingface-cli"),
        "nvidia_smi": shutil.which("nvidia-smi"),
    }
    try:
        checks["disk"] = subprocess.check_output(["df", "-h", str(args.dataset_root.parent)], text=True)
    except Exception as exc:  # noqa: BLE001
        checks["disk_error"] = str(exc)
    try:
        checks["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,compute_cap,power.limit,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        checks["gpu_error"] = str(exc)
    print(json.dumps(checks, indent=2, default=str))


def cmd_ingest_aqua(args: argparse.Namespace) -> None:
    summary = ingest_aqua(
        export_dir=args.export_dir,
        dataset_root=args.dataset_root,
        out_name=args.out_name,
        holdout_percent=args.holdout_percent,
    )
    print(json.dumps(summary, indent=2))


def cmd_ingest_hf(args: argparse.Namespace) -> None:
    if args.config:
        result = install_curated_config(args.config, args.dataset_root, dry_run=args.dry_run)
    else:
        result = download_hf_dataset(
            args.dataset,
            args.dataset_root / "intake",
            allow_patterns=args.allow_pattern,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, indent=2))


def cmd_ingest_yt(args: argparse.Namespace) -> None:
    if args.raw_only:
        result = fetch_youtube_audio(args.url, args.dataset_root / "intake" / "youtube", args.source_id)
    else:
        result = ingest_youtube_video(
            args.url,
            args.dataset_root,
            dataset=args.dataset,
            split=args.split,
            source_id=args.source_id,
            segment_seconds=args.segment_seconds,
            max_segments=args.max_segments,
            domain_tag=args.domain_tag,
            remove_raw_audio=args.remove_raw_audio,
        )
    print(json.dumps(result, indent=2))


def cmd_ingest_yt_batch(args: argparse.Namespace) -> None:
    result = ingest_youtube_candidates(
        candidate_file=args.candidate_file,
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        split=args.split,
        segment_seconds=args.segment_seconds,
        groups=args.group,
        limit=args.limit,
        progress_log=args.progress_log,
        skip_existing=not args.force,
        stop_on_error=args.stop_on_error,
        remove_raw_audio=args.remove_raw_audio,
    )
    print(json.dumps(result, indent=2))


def cmd_label_queue(args: argparse.Namespace) -> None:
    result = make_label_queue(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        split=args.split,
        out=args.out,
        empty=args.empty,
        teacher_file=args.teacher_file,
        scored=args.scored,
        limit=args.limit,
        source_id=args.source_id,
        only_with_teachers=args.only_with_teachers,
    )
    print(json.dumps(result, indent=2))


def cmd_label_teachers(args: argparse.Namespace) -> None:
    result = generate_teacher_transcripts(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        split=args.split,
        teacher_specs=args.teacher,
        out=args.out,
        source_id=args.source_id,
        offset=args.offset,
        limit=args.limit,
        batch_size=args.batch_size,
        force=args.force,
        update_dataset=args.update_dataset,
    )
    print(json.dumps(result, indent=2))


def cmd_label_apply_teachers(args: argparse.Namespace) -> None:
    result = merge_teacher_sidecar_into_dataset(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        teacher_file=args.teacher_file,
    )
    print(json.dumps(result, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    metrics = evaluate_dataset(
        adapter=args.adapter,
        model=args.model,
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        split=args.split,
        out=args.out,
        batch_size=args.batch_size,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        request_timeout=args.request_timeout,
        predictions=args.predictions,
        offset=args.offset,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2))


def cmd_eval_pack(args: argparse.Namespace) -> None:
    result = export_eval_pack(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        split=args.split,
        out_dir=args.out,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2))


def cmd_report_dashboard(args: argparse.Namespace) -> None:
    result = build_report(
        metric_patterns=args.metrics,
        queue_file=args.queue,
        out=args.out,
        json_out=args.json_out,
    )
    print(
        json.dumps(
            {
                "html": result["outputs"]["html"],
                "json": result["outputs"]["json"],
                "metric_runs": len(result["metrics"]),
                "teacher_queue_rows": result["teacher_queue"]["rows"],
            },
            indent=2,
        )
    )


def cmd_report_review_ui(args: argparse.Namespace) -> None:
    result = build_review_ui(
        queue_file=args.queue,
        out=args.out,
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        json_out=args.json_out,
        copy_audio=not args.no_copy_audio,
        title=args.title,
        dedupe=not args.no_dedupe,
    )
    print(json.dumps(result, indent=2))


def cmd_vendor_models(args: argparse.Namespace) -> None:
    result = list_openai_compatible_models(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        expect=args.expect,
        contains=args.contains,
        request_timeout=args.request_timeout,
    )
    print(json.dumps(result, indent=2))


def cmd_repair_text(args: argparse.Namespace) -> None:
    resolver = RepoVocabResolver.from_vocab_files(args.vocab)
    result = resolver.repair(
        args.text,
        mode=args.mode,
        apply_exact=not args.no_exact,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.final_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonon")
    add_common_dataset_root(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor")
    add_subcommand_dataset_root(doctor)
    doctor.set_defaults(func=cmd_doctor)

    ingest = sub.add_parser("ingest")
    ingest_sub = ingest.add_subparsers(dest="ingest_command", required=True)

    aqua = ingest_sub.add_parser("aqua")
    add_subcommand_dataset_root(aqua)
    aqua.add_argument("--export-dir", type=Path, required=True)
    aqua.add_argument("--out-name", default="aqua_voice_seed_v0")
    aqua.add_argument("--holdout-percent", type=int, default=20)
    aqua.set_defaults(func=cmd_ingest_aqua)

    hf = ingest_sub.add_parser("hf")
    add_subcommand_dataset_root(hf)
    hf.add_argument("--dataset")
    hf.add_argument("--allow-pattern", action="append")
    hf.add_argument("--config", type=Path)
    hf.add_argument("--dry-run", action="store_true")
    hf.set_defaults(func=cmd_ingest_hf)

    yt = ingest_sub.add_parser("yt")
    add_subcommand_dataset_root(yt)
    yt.add_argument("--url", required=True)
    yt.add_argument("--source-id")
    yt.add_argument("--dataset", default="youtube_technical_v0")
    yt.add_argument("--split", default="yt_technical_hidden_v0")
    yt.add_argument("--segment-seconds", type=int, default=30)
    yt.add_argument("--max-segments", type=int)
    yt.add_argument("--domain-tag", action="append")
    yt.add_argument("--remove-raw-audio", action="store_true")
    yt.add_argument("--raw-only", action="store_true")
    yt.set_defaults(func=cmd_ingest_yt)

    yt_batch = ingest_sub.add_parser("yt-batch")
    add_subcommand_dataset_root(yt_batch)
    yt_batch.add_argument("--candidate-file", type=Path, required=True)
    yt_batch.add_argument("--dataset", default="youtube_technical_v0")
    yt_batch.add_argument("--split", default="yt_technical_hidden_v0")
    yt_batch.add_argument("--segment-seconds", type=int, default=30)
    yt_batch.add_argument("--group", action="append")
    yt_batch.add_argument("--limit", type=int)
    yt_batch.add_argument("--progress-log", type=Path)
    yt_batch.add_argument("--force", action="store_true")
    yt_batch.add_argument("--stop-on-error", action="store_true")
    yt_batch.add_argument("--remove-raw-audio", action="store_true")
    yt_batch.set_defaults(func=cmd_ingest_yt_batch)

    label = sub.add_parser("label")
    label_sub = label.add_subparsers(dest="label_command", required=True)
    queue = label_sub.add_parser("queue")
    add_subcommand_dataset_root(queue)
    queue.add_argument("--dataset", default="aqua_voice_seed_v0")
    queue.add_argument("--split", required=True)
    queue.add_argument("--out", type=Path)
    queue.add_argument("--empty", action="store_true")
    queue.add_argument("--teacher-file", type=Path)
    queue.add_argument("--scored", action="store_true")
    queue.add_argument("--limit", type=int)
    queue.add_argument("--source-id")
    queue.add_argument("--only-with-teachers", action="store_true")
    queue.set_defaults(func=cmd_label_queue)

    teachers = label_sub.add_parser("teachers")
    add_subcommand_dataset_root(teachers)
    teachers.add_argument("--dataset", default="youtube_technical_v0")
    teachers.add_argument("--split", required=True)
    teachers.add_argument("--teacher", action="append", required=True)
    teachers.add_argument("--out", type=Path)
    teachers.add_argument("--source-id")
    teachers.add_argument("--offset", type=int, default=0)
    teachers.add_argument("--limit", type=int)
    teachers.add_argument("--batch-size", type=int, default=8)
    teachers.add_argument("--force", action="store_true")
    teachers.add_argument("--update-dataset", action="store_true")
    teachers.set_defaults(func=cmd_label_teachers)

    apply_teachers = label_sub.add_parser("apply-teachers")
    add_subcommand_dataset_root(apply_teachers)
    apply_teachers.add_argument("--dataset", default="youtube_technical_v0")
    apply_teachers.add_argument("--teacher-file", type=Path, required=True)
    apply_teachers.set_defaults(func=cmd_label_apply_teachers)

    eval_parser = sub.add_parser("eval")
    add_subcommand_dataset_root(eval_parser)
    eval_parser.add_argument(
        "--adapter",
        choices=["nemo", "whisper", "transformers-asr", "openai-compatible", "external"],
        required=True,
    )
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--dataset", default="aqua_voice_seed_v0")
    eval_parser.add_argument("--split", required=True)
    eval_parser.add_argument("--out", type=Path, required=True)
    eval_parser.add_argument("--batch-size", type=int, default=4)
    eval_parser.add_argument("--predictions", type=Path)
    eval_parser.add_argument("--base-url")
    eval_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    eval_parser.add_argument("--request-timeout", type=float, default=120.0)
    eval_parser.add_argument("--offset", type=int, default=0)
    eval_parser.add_argument("--limit", type=int)
    eval_parser.set_defaults(func=cmd_eval)

    eval_pack = sub.add_parser("eval-pack")
    add_subcommand_dataset_root(eval_pack)
    eval_pack.add_argument("--dataset", default="aqua_voice_seed_v0")
    eval_pack.add_argument("--split", required=True)
    eval_pack.add_argument("--out", type=Path, required=True)
    eval_pack.add_argument("--limit", type=int)
    eval_pack.set_defaults(func=cmd_eval_pack)

    vendor = sub.add_parser("vendor")
    vendor_sub = vendor.add_subparsers(dest="vendor_command", required=True)
    models = vendor_sub.add_parser("models")
    models.add_argument("--base-url", default="https://api.aquavoice.com/api/v1")
    models.add_argument("--api-key-env", default="AQUA_API_KEY")
    models.add_argument("--expect", action="append", default=["avalon-v1.5"])
    models.add_argument("--contains", default="avalon")
    models.add_argument("--request-timeout", type=float, default=30.0)
    models.set_defaults(func=cmd_vendor_models)

    repair = sub.add_parser("repair")
    repair_sub = repair.add_subparsers(dest="repair_command", required=True)
    repair_text = repair_sub.add_parser("text")
    repair_text.add_argument("--text", required=True)
    repair_text.add_argument("--vocab", type=Path, action="append", default=[])
    repair_text.add_argument("--mode", choices=["broad", "gated", "oracle"], default="gated")
    repair_text.add_argument("--no-exact", action="store_true")
    repair_text.add_argument("--json", action="store_true")
    repair_text.set_defaults(func=cmd_repair_text)

    report = sub.add_parser("report")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    dashboard = report_sub.add_parser("dashboard")
    dashboard.add_argument("--metrics", action="append", default=["runs/**/*.metrics.json"])
    dashboard.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_DATASET_ROOT
        / "labels"
        / "queues"
        / "youtube_technical_v0_yt_technical_hidden_v0_teacher_scored_top120.jsonl",
    )
    dashboard.add_argument("--out", type=Path, default=Path("runs/reports/phonon_dashboard.html"))
    dashboard.add_argument("--json-out", type=Path, default=Path("runs/reports/phonon_dashboard.json"))
    dashboard.set_defaults(func=cmd_report_dashboard)

    review_ui = report_sub.add_parser("review-ui")
    add_subcommand_dataset_root(review_ui)
    review_ui.add_argument("--queue", type=Path, required=True)
    review_ui.add_argument("--dataset", default="youtube_technical_v0")
    review_ui.add_argument("--split", default="yt_technical_hidden_v0")
    review_ui.add_argument("--limit", type=int, default=300)
    review_ui.add_argument("--out", type=Path, default=Path("runs/reports/label_review/index.html"))
    review_ui.add_argument("--json-out", type=Path)
    review_ui.add_argument("--title", default="Phonon Label Review")
    review_ui.add_argument("--no-copy-audio", action="store_true")
    review_ui.add_argument("--no-dedupe", action="store_true")
    review_ui.set_defaults(func=cmd_report_review_ui)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
