"""Logging helpers for configuring process-wide Loguru sinks."""

from __future__ import annotations

import multiprocessing
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger


def _resolve_worker_id_from_process_name(process_name: str) -> str:
    """Map process names to stable worker ids for log prefixing."""
    merge_worker_match = re.search(r"worker_\d{2}", process_name)
    if merge_worker_match:
        return merge_worker_match.group(0)

    worker_match = re.fullmatch(r"sciantist-(\d{2})", process_name)
    if worker_match:
        return f"worker_{worker_match.group(1)}"

    return "main"


def _inject_worker_id(record: dict[str, Any]) -> None:
    """Ensure each log record has a worker_id in record extras."""
    process_name = multiprocessing.current_process().name
    record["extra"]["worker_id"] = _resolve_worker_id_from_process_name(process_name)


def _write_per_worker_log(output_dir: Path, message: Any) -> None:
    """Write a log line into workers/<worker_id>/output.log for worker records only."""
    worker_id = str(message.record["extra"].get("worker_id", "")).strip()
    if not worker_id.startswith("worker_"):
        return

    worker_log_dir = output_dir / "workers" / worker_id
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    worker_log_path = worker_log_dir / "output.log"
    with worker_log_path.open("a", encoding="utf-8") as handle:
        handle.write(str(message))


def configure_logging(output_dir: Path, verbose: bool) -> Path:
    """Configure Loguru sinks for console, global output.log, and per-worker output logs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "output.log"
    level = "DEBUG" if verbose else "INFO"

    logger.configure(patcher=_inject_worker_id)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | "
            "<level>{level: <8}</level> | [{extra[worker_id]}] {message}"
        ),
        colorize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_path,
        level=level,
        format="{time:YYYY-MM-DDTHH:mm:ss.SSS!UTC}Z | {level: <8} | [{extra[worker_id]}] {message}",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        lambda message: _write_per_worker_log(output_dir, message),
        level=level,
        format="{time:YYYY-MM-DDTHH:mm:ss.SSS!UTC}Z | {level: <8} | [{extra[worker_id]}] {message}",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    return log_path
