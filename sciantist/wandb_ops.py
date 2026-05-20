"""Weights & Biases run snapshot operations and CLI entrypoint."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

DEFAULT_PROJECT = "dzimmererdkfz-dkfz-german-cancer-research-center/chexclip-pretraining"
DEFAULT_RUN_NAME = "iv0yx2b9"


def _tail_text(text: str, max_chars: int = 8000) -> str:
    """Return trailing text for concise logging/prompts."""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _get_wandb_module() -> Any:
    """Import the wandb module lazily and fail with a helpful message."""
    try:
        return importlib.import_module("wandb")
    except ImportError as error:
        raise RuntimeError("The 'wandb' package is required. Install it with: uv add wandb") from error


def _parse_project_path(project: str, entity: str) -> str:
    """Build a valid W&B project path in the format entity/project."""
    project = project.strip()
    if "/" in project:
        return project

    normalized_entity = entity.strip() or os.getenv("WANDB_ENTITY", "").strip()
    if not normalized_entity:
        raise RuntimeError(
            "Entity is required when project is not in 'entity/project' format. Use --entity or set WANDB_ENTITY."
        )
    return f"{normalized_entity}/{project}"


def _find_run_by_name(
    api: Any,
    project_path: str,
    run_name: str,
    max_wandb_retries: int = 20,
    initial_retry_timeout_seconds: int = 2,
) -> Any:
    """Find a run by display name, internal name, or id within a project."""

    def _is_transient_wandb_error(error: Exception) -> bool:
        """Identify retryable W&B transport/timeout failures."""
        message = str(error).lower()
        error_type_name = type(error).__name__.lower()
        retry_tokens = [
            "timed out",
            "timeout",
            "connection aborted",
            "connection reset",
            "temporarily unavailable",
            "httpsconnectionpool",
            "api.wandb.ai",
        ]
        return error_type_name == "commerror" or any(token in message for token in retry_tokens)

    max_retry_count = max(0, int(max_wandb_retries))
    base_timeout_seconds = max(1, int(initial_retry_timeout_seconds))
    normalized_target = run_name.strip()

    for attempt in range(max_retry_count + 1):
        try:
            candidates = api.runs(project_path)
            for run in candidates:
                display_name = getattr(run, "display_name", "")
                name = getattr(run, "name", "")
                run_id = getattr(run, "id", "")
                if normalized_target in {display_name, name, run_id}:
                    return run
        except Exception as error:
            should_retry = attempt < max_retry_count and _is_transient_wandb_error(error)
            if not should_retry:
                raise
            timeout_seconds = min(600, base_timeout_seconds * (2**attempt))
            time.sleep(timeout_seconds)
            continue

    raise RuntimeError(
        f"No run named '{run_name}' found in project '{project_path}'. "
        "Match is checked against display_name, name, and id."
    )


def _as_finite_float(value: Any) -> float | None:
    """Return value as finite float, otherwise None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _get_last_history_row_and_metric_histories(
    run: Any,
    metric_names: list[str] | None = None,
    max_wandb_retries: int = 20,
    initial_retry_timeout_seconds: int = 2,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Scan full metric history and return final row plus full per-step metric values."""

    def _is_transient_wandb_error(error: Exception) -> bool:
        """Identify retryable W&B transport/timeout failures."""
        message = str(error).lower()
        error_type_name = type(error).__name__.lower()
        retry_tokens = [
            "timed out",
            "timeout",
            "connection aborted",
            "connection reset",
            "temporarily unavailable",
            "httpsconnectionpool",
            "api.wandb.ai",
        ]
        return error_type_name == "commerror" or any(token in message for token in retry_tokens)

    max_retry_count = max(0, int(max_wandb_retries))
    base_timeout_seconds = max(1, int(initial_retry_timeout_seconds))

    best_row: dict[str, Any] | None = None
    best_step = -1
    selected_metrics = {
        metric_name.strip()
        for metric_name in (metric_names or [])
        if isinstance(metric_name, str) and metric_name.strip()
    }
    collect_val_metrics = not selected_metrics

    for attempt in range(max_retry_count + 1):
        metric_histories: dict[str, list[float]] = {metric_name: [] for metric_name in selected_metrics}
        best_row = None
        best_step = -1
        try:
            for row in run.scan_history():
                step = row.get("_step")
                if isinstance(step, int) and step >= best_step:
                    best_step = step
                    best_row = row

                if collect_val_metrics:
                    metric_keys = [key for key in row.keys() if isinstance(key, str) and key.startswith("val/")]
                    for metric_name in metric_keys:
                        numeric_value = _as_finite_float(row.get(metric_name))
                        if numeric_value is None:
                            continue
                        metric_histories.setdefault(metric_name, []).append(numeric_value)
                else:
                    for metric_name in selected_metrics:
                        numeric_value = _as_finite_float(row.get(metric_name))
                        if numeric_value is None:
                            continue
                        metric_histories[metric_name].append(numeric_value)
        except Exception as error:
            should_retry = attempt < max_retry_count and _is_transient_wandb_error(error)
            if not should_retry:
                raise
            timeout_seconds = min(600, base_timeout_seconds * (2**attempt))
            time.sleep(timeout_seconds)
            continue

        if best_row is None:
            return {}, metric_histories
        return best_row, metric_histories

    return {}, {metric_name: [] for metric_name in selected_metrics}


def _download_log_files(run: Any, output_dir: Path) -> list[str]:
    """Download console logs and return local paths."""

    def _is_transient_wandb_error(error: Exception) -> bool:
        """Return True for retryable W&B transport/timeout failures."""
        message = str(error).lower()
        error_type_name = type(error).__name__.lower()
        retry_tokens = [
            "timed out",
            "timeout",
            "connection aborted",
            "connection reset",
            "temporarily unavailable",
            "httpsconnectionpool",
            "api.wandb.ai",
        ]
        return error_type_name == "commerror" or any(token in message for token in retry_tokens)

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []

    # Preferred path from W&B docs for run console logs.
    try:
        output_log = run.file("output.log")
        local_output_log = Path(output_log.download(root=str(output_dir), replace=True).name)
        downloaded.append(str(local_output_log.resolve()))
    except Exception:
        pass

    run_files_iterable: list[Any] = []
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            run_files_iterable = list(run.files())
            break
        except Exception as error:
            should_retry = attempt < (max_attempts - 1) and _is_transient_wandb_error(error)
            if not should_retry:
                return downloaded
            time.sleep(2**attempt)

    for run_file in run_files_iterable:
        file_name = getattr(run_file, "name", "")
        if not file_name.endswith(".log"):
            continue

        try:
            local_path = Path(run_file.download(root=str(output_dir), replace=True).name).resolve()
        except Exception:
            continue
        local_path_str = str(local_path)
        if local_path_str not in downloaded:
            downloaded.append(local_path_str)

    return downloaded


def _load_output_log_excerpt_from_payload(wandb_payload: dict[str, Any], max_chars: int = 8000) -> str:
    """Load output.log text from downloaded snapshot files and return a tail excerpt."""
    log_paths_raw = wandb_payload.get("downloaded_log_files")
    if not isinstance(log_paths_raw, list):
        return ""

    candidate_paths: list[Path] = []
    fallback_paths: list[Path] = []
    for item in log_paths_raw:
        if not isinstance(item, str) or not item.strip():
            continue
        path = Path(item)
        if path.name.lower() == "output.log":
            candidate_paths.append(path)
        elif path.suffix.lower() == ".log":
            fallback_paths.append(path)

    for path in candidate_paths + fallback_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if text.strip():
            return _tail_text(text, max_chars=max_chars)
    return ""


def _to_json_safe(value: Any) -> Any:
    """Recursively convert SDK-specific values into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]

    # wandb old summary containers often keep raw values in a private _dict.
    try:
        internal_dict = object.__getattribute__(value, "_dict")
    except Exception:
        internal_dict = None
    if isinstance(internal_dict, dict):
        return {str(key): _to_json_safe(item) for key, item in internal_dict.items()}

    try:
        dict_method = object.__getattribute__(value, "to_dict")
    except Exception:
        dict_method = None
    if callable(dict_method):
        try:
            return _to_json_safe(dict_method())
        except Exception:
            pass

    try:
        item_method = object.__getattribute__(value, "item")
    except Exception:
        item_method = None
    if callable(item_method):
        try:
            return _to_json_safe(item_method())
        except Exception:
            pass

    return str(value)


def _build_payload(
    run: Any,
    last_row: dict[str, Any],
    metric_histories: dict[str, list[float]],
    log_paths: list[str],
) -> dict[str, Any]:
    """Create a JSON-serializable output payload for the selected run."""
    return {
        "project": run.project,
        "entity": run.entity,
        "run_id": run.id,
        "run_name": run.name,
        "run_display_name": getattr(run, "display_name", None),
        "run_path": "/".join(run.path),
        "state": run.state,
        "url": run.url,
        "summary": _to_json_safe(dict(run.summary)),
        "last_step_metrics": _to_json_safe(last_row),
        "metric_histories": _to_json_safe(metric_histories),
        "downloaded_log_files": _to_json_safe(log_paths),
    }


def _extract_avg_gpu_metrics_from_events(events_table: Any) -> tuple[float | None, float | None]:
    """Compute average GPU utilization and memory utilization percentage from W&B events."""

    def _as_rows(table: Any) -> list[dict[str, Any]]:
        if table is None:
            return []
        if isinstance(table, list):
            return [row for row in table if isinstance(row, dict)]
        to_dict = getattr(table, "to_dict", None)
        if callable(to_dict):
            try:
                rows = to_dict(orient="records")
                if isinstance(rows, list):
                    return [row for row in rows if isinstance(row, dict)]
            except Exception:
                return []
        return []

    rows = _as_rows(events_table)
    if not rows:
        return None, None

    util_pattern = re.compile(r"(gpu.*util|utilization.*gpu|system\.gpu\.\d+\.gpu|gpu\.\d+\.util)", re.IGNORECASE)
    mem_pct_pattern = re.compile(
        r"(memory.*(percent|pct|util)|mem.*(percent|pct|util)|memoryallocated.*(%|percent|pct)|gpu.*memory.*(percent|pct|util))",
        re.IGNORECASE,
    )
    mem_alloc_pattern = re.compile(
        r"(memoryallocated|memallocated|memory_used|memoryused|fb_memory_usage|gpu.*memory.*used)",
        re.IGNORECASE,
    )
    mem_total_pattern = re.compile(
        r"(memorytotal|memtotal|memory_total|total_memory|gpu.*memory.*total)",
        re.IGNORECASE,
    )
    gpu_id_pattern = re.compile(r"gpu[\._](\d+)", re.IGNORECASE)

    def _normalize_percent(value: float) -> float:
        if 0.0 <= value <= 1.0:
            return value * 100.0
        return value

    util_values: list[float] = []
    mem_values_pct: list[float] = []
    for row in rows:
        alloc_by_gpu: dict[str, float] = {}
        total_by_gpu: dict[str, float] = {}
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            lower_key = str(key).lower()
            if lower_key.startswith("_"):
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue

            if util_pattern.search(lower_key) and ("memory" not in lower_key):
                util_values.append(numeric_value)
                continue

            if mem_pct_pattern.search(lower_key):
                mem_values_pct.append(_normalize_percent(numeric_value))
                continue

            gpu_match = gpu_id_pattern.search(lower_key)
            gpu_id = gpu_match.group(1) if gpu_match else "global"
            if mem_alloc_pattern.search(lower_key) and not mem_total_pattern.search(lower_key):
                alloc_by_gpu[gpu_id] = numeric_value
            elif mem_total_pattern.search(lower_key):
                total_by_gpu[gpu_id] = numeric_value

        for gpu_id, alloc in alloc_by_gpu.items():
            total = total_by_gpu.get(gpu_id)
            if total and total > 0.0:
                mem_values_pct.append((alloc / total) * 100.0)

    avg_util = (sum(util_values) / len(util_values)) if util_values else None
    avg_mem = (sum(mem_values_pct) / len(mem_values_pct)) if mem_values_pct else None
    return avg_util, avg_mem


class ExperimentTrackingBackend(ABC):
    """Contract for fetching run snapshots and logs from tracking systems."""

    @abstractmethod
    def sync_from_cluster(self, log_fn: Callable[[str], None] | None = None) -> None:
        """Sync remote tracking data into local storage before querying runs."""

    @abstractmethod
    def fetch_run_snapshot_by_job_id(
        self,
        project_path: str,
        job_id: str,
        output_dir: Path,
        metric_names: list[str] | None = None,
        max_wandb_retries: int = 20,
    ) -> dict[str, Any]:
        """Fetch one run snapshot and persist a local JSON payload."""

    @abstractmethod
    def load_output_log_excerpt(self, payload: dict[str, Any], max_chars: int = 8000) -> str:
        """Load a concise output.log excerpt from a snapshot payload."""

    @abstractmethod
    def fetch_avg_gpu_metrics_from_events(self, project_path: str, job_id: str) -> tuple[float | None, float | None]:
        """Compute average GPU utilization and memory utilization from tracker events."""


class WandbTrackingBackend(ExperimentTrackingBackend):
    """Experiment-tracking backend backed by Weights & Biases."""

    def sync_from_cluster(self, log_fn: Callable[[str], None] | None = None) -> None:
        """Run the configured W&B sync command before querying run data."""
        _sync_wandb_from_cluster(log_fn=log_fn)

    def fetch_run_snapshot_by_job_id(
        self,
        project_path: str,
        job_id: str,
        output_dir: Path,
        metric_names: list[str] | None = None,
        max_wandb_retries: int = 20,
    ) -> dict[str, Any]:
        """Fetch and persist one W&B run snapshot using the run id/job id."""
        if not project_path.strip():
            raise RuntimeError("W&B project path is required (entity/project).")

        output_dir.mkdir(parents=True, exist_ok=True)
        wandb_module = _get_wandb_module()
        api = wandb_module.Api(timeout=120)
        run = _find_run_by_name(
            api,
            project_path,
            job_id,
            max_wandb_retries=max(0, int(max_wandb_retries)),
        )
        last_row, metric_histories = _get_last_history_row_and_metric_histories(
            run,
            metric_names=metric_names,
            max_wandb_retries=max(0, int(max_wandb_retries)),
        )
        logs_dir = output_dir / "logs" / str(run.id)
        log_paths = _download_log_files(run, logs_dir)
        payload = _build_payload(run, last_row, metric_histories, log_paths)
        json_path = output_dir / f"{run.id}_snapshot.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return payload

    def load_output_log_excerpt(self, payload: dict[str, Any], max_chars: int = 8000) -> str:
        """Load output.log text from a payload and return a concise tail."""
        return _load_output_log_excerpt_from_payload(payload, max_chars=max_chars)

    def fetch_avg_gpu_metrics_from_events(self, project_path: str, job_id: str) -> tuple[float | None, float | None]:
        """Read event history and compute average GPU metrics for one run."""
        try:
            wandb_module = _get_wandb_module()
            api = wandb_module.Api(timeout=120)
            run = _find_run_by_name(api, project_path, job_id)
            events_table = run.history(stream="events")
            return _extract_avg_gpu_metrics_from_events(events_table)
        except Exception:
            return None, None


def create_tracking_backend(backend_name: str) -> ExperimentTrackingBackend:
    """Create an experiment-tracking backend from a configured backend name."""
    normalized_name = backend_name.strip().lower()
    if normalized_name == "wandb":
        return WandbTrackingBackend()
    raise ValueError(f"Unsupported tracking backend: {backend_name!r}")


def _sync_wandb_from_cluster(log_fn: Callable[[str], None] | None = None) -> None:
    """Run the clustermin W&B sync script before pulling run snapshots."""
    sync_cwd = "/home/zimmerer/ws/clustermin"
    sync_command = [
        "/home/zimmerer/ws/clustermin/.venv/bin/python",
        "/home/zimmerer/ws/clustermin/ssh_sync/sync_wandb.py",
    ]
    if log_fn is not None:
        log_fn(f"Running W&B sync command: {' '.join(sync_command)} (cwd={sync_cwd})")
    subprocess.run(sync_command, cwd=sync_cwd, check=True, text=True, capture_output=True)


def pull_run_snapshot(
    project: str = DEFAULT_PROJECT,
    run_name: str = DEFAULT_RUN_NAME,
    entity: str = "",
    output_dir: str | Path = "outputs/wandb_pull",
    max_wandb_retries: int = 20,
) -> tuple[dict[str, Any], Path]:
    """Fetch run metrics/logs, persist a snapshot JSON, and return payload plus output path."""
    project_path = _parse_project_path(project, entity)
    normalized_output_dir = Path(output_dir)
    backend = WandbTrackingBackend()
    payload = backend.fetch_run_snapshot_by_job_id(
        project_path=project_path,
        job_id=run_name,
        output_dir=normalized_output_dir,
        max_wandb_retries=max(0, int(max_wandb_retries)),
    )
    run_id = str(payload.get("run_id", run_name))
    json_path = normalized_output_dir / f"{run_id}_snapshot.json"

    return payload, json_path


def main() -> None:
    """Parse CLI args, fetch run data, and write outputs to disk."""
    parser = argparse.ArgumentParser(
        description=("Pull W&B data for one run: final-step metrics and available log files.")
    )
    parser.add_argument(
        "project",
        nargs="?",
        default=DEFAULT_PROJECT,
        help=(f"Project name or path (entity/project). Default: {DEFAULT_PROJECT}"),
    )
    parser.add_argument(
        "run_name",
        nargs="?",
        default=DEFAULT_RUN_NAME,
        help=(f"Run display name, name, or run id. Default: {DEFAULT_RUN_NAME}"),
    )
    parser.add_argument(
        "--entity",
        default="",
        help="W&B entity/team; required when project is not entity/project",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/wandb_pull",
        help="Directory where logs and JSON output are written",
    )
    parser.add_argument(
        "--max-wandb-retries",
        type=int,
        default=20,
        help="Maximum retries for transient W&B API errors",
    )

    args = parser.parse_args()

    payload, json_path = pull_run_snapshot(
        project=args.project,
        run_name=args.run_name,
        entity=args.entity,
        output_dir=args.output_dir,
        max_wandb_retries=max(0, int(args.max_wandb_retries)),
    )

    print(f"Run: {payload['run_path']}")
    print(f"Snapshot JSON: {json_path.resolve()}")
    if payload["downloaded_log_files"]:
        print("Downloaded log files:")
        for log_path in payload["downloaded_log_files"]:
            print(f"- {log_path}")
    else:
        print("No .log files were found for this run.")


__all__ = [
    "DEFAULT_PROJECT",
    "DEFAULT_RUN_NAME",
    "ExperimentTrackingBackend",
    "WandbTrackingBackend",
    "create_tracking_backend",
    "main",
    "pull_run_snapshot",
]
