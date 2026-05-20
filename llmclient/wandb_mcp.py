"""MCP server for querying Weights & Biases runs and artifacts."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Keep downloaded artifacts in a predictable location.
DOWNLOAD_ROOT = Path(os.getenv("WANDB_MCP_DOWNLOAD_DIR", "outputs/wandb_mcp")).resolve()

mcp = FastMCP("WandB Run Explorer")


def _get_wandb_module() -> Any:
    """Import wandb lazily to keep startup lightweight."""
    try:
        return importlib.import_module("wandb")
    except ImportError as error:
        raise RuntimeError("The 'wandb' package is required. Install it with: uv add wandb") from error


def _normalize_project_path(project_name: str, entity: str = "") -> str:
    """Resolve project identifier to entity/project format."""
    normalized = project_name.strip()
    if not normalized:
        raise ValueError("project_name must not be empty")
    if "/" in normalized:
        return normalized

    normalized_entity = entity.strip() or os.getenv("WANDB_ENTITY", "").strip()
    if not normalized_entity:
        raise ValueError(
            "project_name must be 'entity/project' or an entity must be provided via the entity argument or WANDB_ENTITY"
        )
    return f"{normalized_entity}/{normalized}"


def _to_json_safe(value: Any) -> Any:
    """Convert SDK-specific values into JSON primitives recursively."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]

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

    return str(value)


def _get_api_and_run(project_name: str, run_id: str, entity: str = "") -> tuple[Any, Any, str]:
    """Return initialized API and selected run."""
    wandb = _get_wandb_module()
    api = wandb.Api(timeout=120)
    project_path = _normalize_project_path(project_name=project_name, entity=entity)

    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must not be empty")

    run = api.run(f"{project_path}/{normalized_run_id}")
    return api, run, project_path


def _run_info_payload(run: Any) -> dict[str, Any]:
    """Build a compact payload for common run metadata."""
    return {
        "project": run.project,
        "entity": run.entity,
        "run_id": run.id,
        "run_name": run.name,
        "run_display_name": getattr(run, "display_name", None),
        "run_path": "/".join(run.path),
        "state": run.state,
        "url": run.url,
        "created_at": getattr(run, "created_at", None),
        "heartbeat_at": getattr(run, "heartbeat_at", None),
        "summary": _to_json_safe(dict(run.summary)),
        "config": _to_json_safe(dict(run.config)),
    }


@mcp.tool()
def get_run_info(project_name: str, run_id: str, entity: str = "") -> dict[str, Any]:
    """Get run metadata, config, and summary metrics."""
    try:
        _, run, _ = _get_api_and_run(project_name=project_name, run_id=run_id, entity=entity)
        return _run_info_payload(run)
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_run_history(
    project_name: str,
    run_id: str,
    entity: str = "",
    metric_names: list[str] | None = None,
    max_rows: int = 2000,
) -> dict[str, Any]:
    """Get full or filtered row-wise run history (metrics/log values per step)."""
    try:
        _, run, project_path = _get_api_and_run(project_name=project_name, run_id=run_id, entity=entity)

        selected_keys = [key.strip() for key in (metric_names or []) if isinstance(key, str) and key.strip()]
        rows: list[dict[str, Any]] = []
        for row in run.scan_history(keys=selected_keys or None):
            rows.append(_to_json_safe(dict(row)))
            if len(rows) >= max(1, max_rows):
                break

        return {
            "project_path": project_path,
            "run_id": run.id,
            "row_count": len(rows),
            "truncated": len(rows) >= max(1, max_rows),
            "rows": rows,
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_run_metric_series(
    project_name: str,
    run_id: str,
    metric_names: list[str],
    entity: str = "",
    max_rows: int = 5000,
) -> dict[str, Any]:
    """Get per-metric value series across history for selected metric names."""
    try:
        if not metric_names:
            return {"error": "metric_names must contain at least one metric key"}

        _, run, project_path = _get_api_and_run(project_name=project_name, run_id=run_id, entity=entity)

        cleaned_metric_names = [key.strip() for key in metric_names if isinstance(key, str) and key.strip()]
        series: dict[str, list[float]] = {name: [] for name in cleaned_metric_names}
        scanned_rows = 0

        for row in run.scan_history(keys=cleaned_metric_names):
            scanned_rows += 1
            for name in cleaned_metric_names:
                value = row.get(name)
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    series[name].append(float(value))
            if scanned_rows >= max(1, max_rows):
                break

        return {
            "project_path": project_path,
            "run_id": run.id,
            "scanned_rows": scanned_rows,
            "truncated": scanned_rows >= max(1, max_rows),
            "series": _to_json_safe(series),
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_run_files(project_name: str, run_id: str, entity: str = "") -> dict[str, Any]:
    """List files attached to the run, including sizes and update timestamps."""
    try:
        _, run, project_path = _get_api_and_run(project_name=project_name, run_id=run_id, entity=entity)

        files: list[dict[str, Any]] = []
        for run_file in run.files():
            files.append(
                {
                    "name": getattr(run_file, "name", ""),
                    "size_bytes": getattr(run_file, "size", None),
                    "updated_at": getattr(run_file, "updated_at", None),
                    "url": getattr(run_file, "url", None),
                }
            )

        return {
            "project_path": project_path,
            "run_id": run.id,
            "file_count": len(files),
            "files": files,
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def download_run_files(
    project_name: str,
    run_id: str,
    file_suffixes: list[str] | None = None,
    entity: str = "",
) -> dict[str, Any]:
    """Download all or filtered run files to WANDB_MCP_DOWNLOAD_DIR and return local paths."""
    try:
        _, run, project_path = _get_api_and_run(project_name=project_name, run_id=run_id, entity=entity)

        suffixes = {
            suffix.lower().strip() for suffix in (file_suffixes or []) if isinstance(suffix, str) and suffix.strip()
        }
        download_dir = (DOWNLOAD_ROOT / run.entity / run.project / run.id).resolve()
        download_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[str] = []
        for run_file in run.files():
            name = getattr(run_file, "name", "")
            if suffixes and not any(name.lower().endswith(suffix) for suffix in suffixes):
                continue
            local_file = run_file.download(root=str(download_dir), replace=True)
            local_path = Path(getattr(local_file, "name", "")).resolve()
            downloaded.append(str(local_path))

        return {
            "project_path": project_path,
            "run_id": run.id,
            "download_dir": str(download_dir),
            "downloaded_count": len(downloaded),
            "downloaded_files": downloaded,
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_run_logs(
    project_name: str,
    run_id: str,
    entity: str = "",
    max_chars: int = 30000,
) -> dict[str, Any]:
    """Download log files and return text content (trimmed from the end for very large logs)."""
    try:
        files_result = download_run_files(
            project_name=project_name,
            run_id=run_id,
            file_suffixes=[".log", ".txt", ".jsonl"],
            entity=entity,
        )
        if files_result.get("error"):
            return files_result

        logs: list[dict[str, Any]] = []
        max_chars_normalized = max(1, max_chars)
        for path_str in files_result.get("downloaded_files", []):
            path = Path(path_str)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if len(text) > max_chars_normalized:
                text = text[-max_chars_normalized:]
            logs.append({"path": str(path), "content": text})

        return {
            "project_path": files_result.get("project_path"),
            "run_id": files_result.get("run_id"),
            "log_count": len(logs),
            "logs": logs,
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_run_full_snapshot(
    project_name: str,
    run_id: str,
    entity: str = "",
    history_max_rows: int = 1000,
    log_max_chars: int = 30000,
) -> dict[str, Any]:
    """Return an all-in-one payload with info, history, files, and log excerpts."""
    info = get_run_info(project_name=project_name, run_id=run_id, entity=entity)
    if info.get("error"):
        return info

    history = get_run_history(
        project_name=project_name,
        run_id=run_id,
        entity=entity,
        max_rows=history_max_rows,
    )
    files = list_run_files(project_name=project_name, run_id=run_id, entity=entity)
    logs = get_run_logs(
        project_name=project_name,
        run_id=run_id,
        entity=entity,
        max_chars=log_max_chars,
    )

    payload = {
        "run_info": info,
        "history": history,
        "files": files,
        "logs": logs,
    }

    snapshot_dir = (
        DOWNLOAD_ROOT / info.get("entity", "unknown") / info.get("project", "unknown") / info.get("run_id", run_id)
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "full_snapshot.json"
    snapshot_path.write_text(json.dumps(_to_json_safe(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    payload["snapshot_path"] = str(snapshot_path)
    return payload


if __name__ == "__main__":
    mcp.run()
