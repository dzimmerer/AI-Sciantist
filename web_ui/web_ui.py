"""Flask web UI for Sciantist experiment tracking and visualization."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, make_response, render_template, request

from sciantist.state import _load_json_state


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _safe_jsonify(data: dict):
    """Return JSON response with NaN/Inf safely converted to None."""
    return make_response(
        (json.dumps(_sanitize(data)), 200, {"Content-Type": "application/json"})
    )


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _load_leaderboard(output_dir: Path) -> list[dict[str, Any]]:
    """Load leaderboard entries from JSON."""
    leaderboard_path = output_dir / "leaderboard.json"
    if not leaderboard_path.exists():
        return []
    try:
        payload = json.loads(leaderboard_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _load_worker_states(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load worker states for all workers."""
    workers_dir = output_dir / "workers"
    if not workers_dir.exists():
        return {}
    worker_states = {}
    for worker_path in workers_dir.iterdir():
        if worker_path.is_dir() and worker_path.name.startswith("worker_"):
            state_path = worker_path / "worker_state.json"
            info_path = worker_path / "worker_info.json"
            if state_path.exists():
                worker_id = worker_path.name
                state = _load_json_state(state_path)
                info = _load_json_state(info_path) if info_path.exists() else {}
                worker_states[worker_id] = {
                    **state,
                    "worker_info": info,
                    "state_path": str(state_path),
                }
    return worker_states


def _load_run_details(
    output_dir: Path, worker_id: str, run_index: int
) -> dict[str, Any] | None:
    """Load run details for a specific worker run."""
    run_path = output_dir / "workers" / worker_id / f"run_{run_index:06d}"
    if not run_path.exists():
        return None
    run_info_path = run_path / "run_info.json"
    if run_info_path.exists():
        return _load_json_state(run_info_path)
    return None


def _load_memory(output_dir: Path) -> str:
    """Load project memory markdown."""
    memory_path = output_dir / "memory.md"
    if memory_path.exists():
        return memory_path.read_text(encoding="utf-8")
    return ""


def _to_finite_float(value: Any) -> float | None:
    """Return a finite float metric value, or None when value is invalid."""
    import math

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _count_active_workers(output_dir: Path) -> int:
    """Count workers that have an active non-idle run (running or pending).

    Entries are only appended to the leaderboard when an experiment terminalizes,
    so running experiments are not present in the leaderboard. We count active
    workers as a proxy for in-progress runs.
    """
    workers_dir = output_dir / "workers"
    if not workers_dir.exists():
        return 0
    count = 0
    for worker_path in workers_dir.iterdir():
        if worker_path.is_dir() and worker_path.name.startswith("worker_"):
            state_path = worker_path / "worker_state.json"
            if state_path.exists():
                state = _load_json_state(state_path)
                if state.get("active_run_id") and state.get("worker_stage") != "idle":
                    count += 1
    return count


def _load_experiments_summary(output_dir: Path) -> dict[str, Any]:
    """Load experiments summary from leaderboard data."""
    leaderboard = _load_leaderboard(output_dir)
    if not leaderboard:
        return {
            "total_runs": 0,
            "finished_runs": 0,
            "failed_runs": 0,
            "in_progress_runs": _count_active_workers(output_dir),
            "best_metric": None,
            "avg_metric": None,
            "total_experiments": 0,
            "unique_workers": 0,
            "workers_list": [],
        }

    finished = [e for e in leaderboard if e.get("status") == "finished"]
    metrics = [
        _to_finite_float(e.get("unified_metric"))
        for e in finished
        if _to_finite_float(e.get("unified_metric")) is not None
    ]

    unique_workers = set(
        e.get("worker_id", "") for e in leaderboard if e.get("worker_id")
    )

    return {
        "total_runs": len(leaderboard),
        "finished_runs": len(finished),
        "failed_runs": len([e for e in leaderboard if e.get("status") == "crashed"]),
        "in_progress_runs": _count_active_workers(output_dir),
        "best_metric": max(metrics) if metrics else None,
        "avg_metric": sum(metrics) / len(metrics) if metrics else None,
        "total_experiments": len(finished),
        "unique_workers": len(unique_workers),
        "workers_list": sorted(unique_workers),
    }

    finished = [e for e in leaderboard if e.get("status") == "finished"]
    metrics = [
        _to_finite_float(e.get("unified_metric"))
        for e in finished
        if _to_finite_float(e.get("unified_metric")) is not None
    ]

    unique_workers = set(
        e.get("worker_id", "") for e in leaderboard if e.get("worker_id")
    )

    return {
        "total_runs": len(leaderboard),
        "finished_runs": len(finished),
        "failed_runs": len([e for e in leaderboard if e.get("status") == "crashed"]),
        "in_progress_runs": len(
            [e for e in leaderboard if e.get("status") not in ("finished", "crashed")]
        )
        + _count_running_workers(output_dir),
        "best_metric": max(metrics) if metrics else None,
        "avg_metric": sum(metrics) / len(metrics) if metrics else None,
        "total_experiments": len(finished),
        "unique_workers": len(unique_workers),
        "workers_list": sorted(unique_workers),
    }


def _format_runtime(runtime_seconds: int | None) -> str:
    """Format runtime seconds as DD-HH:MM:SS."""
    if runtime_seconds is None:
        return "unknown"
    total_seconds = max(0, int(runtime_seconds))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_iso_timestamp(ts: str | None) -> datetime | None:
    """Parse ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


_run_info_cache: dict[str, dict[str, Any]] = {}


def _get_run_info(workers_dir: Path, run_id: str) -> dict[str, Any]:
    """Get run_info from cache or by scanning worker directories."""
    if run_id in _run_info_cache:
        return _run_info_cache.get(run_id, {})
    for wp in workers_dir.iterdir():
        if not wp.is_dir() or not wp.name.startswith("worker_"):
            continue
        for rd in wp.iterdir():
            if not rd.is_dir() or not rd.name.startswith("run_"):
                continue
            run_info_path = rd / "run_info.json"
            if run_info_path.exists():
                run_info = _load_json_state(run_info_path)
                exp_outcome = run_info.get("experiment_outcome", {})
                if exp_outcome.get("run_id") == run_id:
                    _run_info_cache[run_id] = exp_outcome
                    return exp_outcome
    return {}


def _iter_active_runs(output_dir: Path) -> list[dict[str, Any]]:
    """Iterate active (non-terminal) runs from worker run_info files.

    These runs have been started but not yet terminalized, so they are not
    in the leaderboard. We scrape them from run_info.json files and
    run_state.json for job_id.
    """
    workers_dir = output_dir / "workers"
    if not workers_dir.exists():
        return []
    active_runs = []
    for worker_path in workers_dir.iterdir():
        if not worker_path.is_dir() or not worker_path.name.startswith("worker_"):
            continue
        worker_id = worker_path.name
        state_path = worker_path / "worker_state.json"
        state = _load_json_state(state_path) if state_path.exists() else {}
        active_run_id = state.get("active_run_id", "")
        worker_stage = state.get("worker_stage", "")
        if not active_run_id or worker_stage == "idle":
            continue
        worker_info_path = worker_path / "worker_info.json"
        worker_info = (
            _load_json_state(worker_info_path) if worker_info_path.exists() else {}
        )
        expert = worker_info.get("expert", {})
        worker_role = (
            expert.get("name", "general") if isinstance(expert, dict) else "general"
        )
        for run_dir in worker_path.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            run_info_path = run_dir / "run_info.json"
            if not run_info_path.exists():
                continue
            run_info = _load_json_state(run_info_path)
            ac = run_info.get("active_cycle", {})
            if ac.get("run_id") != active_run_id:
                continue
            trial_state_path = run_dir / "run_state.json"
            trial_state = (
                _load_json_state(trial_state_path) if trial_state_path.exists() else {}
            )
            idea_payload = run_info.get("idea_payload", {})
            status = "running" if worker_stage == "executing" else "pending"
            outline = (
                idea_payload.get("idea_outline", "")
                or idea_payload.get("rough_outline", "")
                or ac.get("idea_outline", "")
            )
            active_runs.append(
                {
                    "run_id": active_run_id,
                    "worker_id": worker_id,
                    "worker_role": worker_role,
                    "idea_title": idea_payload.get("idea_title", "")
                    or ac.get("idea_title", ""),
                    "idea_outline": outline,
                    "aider_plan_prompt": idea_payload.get("aider_plan_prompt", "")
                    or ac.get("aider_plan_prompt", ""),
                    "aider_impl_prompt": idea_payload.get("aider_impl_prompt", "")
                    or ac.get("aider_impl_prompt", ""),
                    "status": status,
                    "runtime_seconds": trial_state.get("last_runtime_seconds"),
                    "unified_metric": None,
                    "timestamp_utc": trial_state.get("checkpoint_updated_ts_utc", ""),
                    "job_id": trial_state.get("job_id", ""),
                    "worker_stage": worker_stage,
                    "feature_branch": ac.get("feature_branch", ""),
                    "baseline_commit": ac.get("baseline_commit", ""),
                    "trial_commit": trial_state.get("trial_commit", ""),
                    "currently_in_best_path": False,
                    "kept": False,
                }
            )
    return active_runs


def _enrich_entry(entry: dict[str, Any], workers_dir: Path) -> dict[str, Any]:
    """Enrich entry with runtime_seconds and other fields from run_info if available."""
    entry = dict(entry)
    run_id = entry.get("run_id", "")
    if not run_id or not workers_dir.exists():
        return entry
    exp_outcome = _get_run_info(workers_dir, run_id)
    if exp_outcome:
        if (
            entry.get("runtime_seconds") is None
            and exp_outcome.get("runtime_seconds") is not None
        ):
            entry["runtime_seconds"] = exp_outcome["runtime_seconds"]
        if (
            entry.get("avg_gpu_util") is None
            and exp_outcome.get("avg_gpu_util") is not None
        ):
            entry["avg_gpu_util"] = exp_outcome["avg_gpu_util"]
        if (
            entry.get("avg_gpu_memory") is None
            and exp_outcome.get("avg_gpu_memory") is not None
        ):
            entry["avg_gpu_memory"] = exp_outcome["avg_gpu_memory"]
        if (
            entry.get("job_id") is None
            and exp_outcome.get("job_id") is not None
        ):
            entry["job_id"] = exp_outcome["job_id"]
    return entry


def _build_parent_history(
    leaderboard: list[dict[str, Any]],
    run_id: str,
    workers_dir: Path | None = None,
    max_depth: int = 20,
) -> list[dict[str, Any]]:
    """Build parent chain from parent_best_run_id, returning list of ancestors."""
    by_run_id: dict[str, dict[str, Any]] = {e.get("run_id", ""): e for e in leaderboard}

    def _get_entry_and_parent(rid: str) -> tuple[dict[str, Any] | None, str | None]:
        entry = by_run_id.get(rid, {})
        if entry:
            parent_id = (
                entry.get("parent_best_run_id") if isinstance(entry, dict) else None
            )
            if parent_id:
                return entry, parent_id
        parent_from_cycle = None
        cycle_entry = None
        if workers_dir:
            for wp in workers_dir.iterdir():
                if not wp.is_dir() or not wp.name.startswith("worker_"):
                    continue
                for rd in wp.iterdir():
                    if not rd.is_dir() or not rd.name.startswith("run_"):
                        continue
                    run_info_path = rd / "run_info.json"
                    if run_info_path.exists():
                        run_info = _load_json_state(run_info_path)
                        ac = run_info.get("active_cycle", {})
                        if ac.get("run_id") == rid:
                            parent_from_cycle = ac.get("parent_best_run_id")
                            cycle_entry = ac
                            break
                        exp_outcome = run_info.get("experiment_outcome", {})
                        if exp_outcome.get("run_id") == rid:
                            if not cycle_entry:
                                cycle_entry = exp_outcome
                if parent_from_cycle:
                    break
        if cycle_entry and not parent_from_cycle:
            parent_from_cycle = cycle_entry.get("parent_best_run_id")
        return entry if entry else cycle_entry, parent_from_cycle

    history = []
    current_id = run_id
    visited: set[str] = set()

    for _ in range(max_depth):
        if not current_id or current_id in visited:
            break
        visited.add(current_id)

        entry, parent_id = _get_entry_and_parent(current_id)

        if not parent_id:
            break

        parent_entry, _ = (
            _get_entry_and_parent(parent_id) if parent_id in by_run_id else (None, None)
        )
        if not parent_entry:
            parent_entry = by_run_id.get(parent_id, {})

        history.append(
            {
                "run_id": parent_id,
                "worker_id": parent_entry.get("worker_id", "") if parent_entry else "",
                "idea_title": parent_entry.get("idea_title", "")
                if parent_entry
                else "",
                "unified_metric": _to_finite_float(parent_entry.get("unified_metric"))
                if parent_entry
                else None,
                "timestamp_utc": parent_entry.get("timestamp_utc", "")
                if parent_entry
                else "",
                "status": parent_entry.get("status", "") if parent_entry else "",
            }
        )

        current_id = parent_id

    return history


def create_app(output_dir: str = "./outputs/sciantist") -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    resolved_output_dir = Path(output_dir).expanduser().resolve()

    @app.route("/")
    def index() -> str:
        return render_template("index.html", output_dir=str(resolved_output_dir))

    @app.route("/api/overview")
    def api_overview() -> jsonify:
        summary = _load_experiments_summary(resolved_output_dir)
        leaderboard = _load_leaderboard(resolved_output_dir)
        workers_dir = resolved_output_dir / "workers"
        memory = _load_memory(resolved_output_dir)

        recent_runs = sorted(
            leaderboard,
            key=lambda x: _parse_iso_timestamp(x.get("timestamp_utc"))
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )[:10]
        recent_runs = [_enrich_entry(e, workers_dir) for e in recent_runs]

        best_runs = sorted(
            [e for e in leaderboard if e.get("unified_metric") is not None],
            key=lambda x: x.get("unified_metric", 0),
            reverse=True,
        )[:10]
        best_runs = [_enrich_entry(e, workers_dir) for e in best_runs]

        return _safe_jsonify(
            {
                "summary": summary,
                "recent_runs": recent_runs,
                "best_runs": best_runs,
                "memory": memory,
            }
        )

    @app.route("/api/leaderboard")
    def api_leaderboard() -> jsonify:
        leaderboard = _load_leaderboard(resolved_output_dir)
        workers_dir = resolved_output_dir / "workers"
        sort_by = request.args.get("sort", "metric")
        sort_order = request.args.get("order", "desc")

        filtered = [
            _enrich_entry(e, workers_dir)
            for e in leaderboard
            if e.get("status") == "finished"
        ]

        if sort_by == "metric":
            filtered.sort(
                key=lambda x: x.get("unified_metric")
                if x.get("unified_metric") is not None
                else float("-inf"),
                reverse=(sort_order == "desc"),
            )
        elif sort_by == "runtime":
            filtered.sort(
                key=lambda x: x.get("runtime_seconds")
                if x.get("runtime_seconds") is not None
                else float("inf"),
                reverse=(sort_order == "desc"),
            )
        elif sort_by == "timestamp":
            filtered.sort(
                key=lambda x: _parse_iso_timestamp(x.get("timestamp_utc"))
                or datetime.min,
                reverse=(sort_order == "desc"),
            )
        elif sort_by == "worker":
            filtered.sort(
                key=lambda x: x.get("worker_id", ""),
                reverse=(sort_order == "desc"),
            )

        return _safe_jsonify({"leaderboard": filtered, "total": len(filtered)})

    @app.route("/api/experiments")
    def api_experiments() -> jsonify:
        leaderboard = _load_leaderboard(resolved_output_dir)
        workers_dir = resolved_output_dir / "workers"
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
        status_filter = request.args.get("status")
        worker_filter = request.args.get("worker")
        sort_by = request.args.get("sort", "timestamp_utc")
        sort_order = request.args.get("order", "desc")

        all_runs = list(leaderboard) + _iter_active_runs(resolved_output_dir)
        filtered = [_enrich_entry(e, workers_dir) for e in all_runs]

        if status_filter and status_filter != "all":
            filtered = [e for e in filtered if e.get("status") == status_filter]

        if worker_filter:
            filtered = [e for e in filtered if worker_filter in e.get("worker_id", "")]

        _STATUS_PRIORITY = {"running": 0, "pending": 1, "finished": 2, "crashed": 3}

        def _make_sort_value(x: dict[str, Any]) -> Any:
            status = x.get("status", "")
            if sort_by == "status":
                return _STATUS_PRIORITY.get(status, 99)
            elif sort_by == "unified_metric":
                v = x.get("unified_metric")
                return v if v is not None else float("-inf")
            elif sort_by == "metric_delta":
                v = x.get("metric_delta")
                return v if v is not None else float("-inf")
            elif sort_by == "runtime_seconds":
                v = x.get("runtime_seconds")
                return v if v is not None else 0
            elif sort_by == "kept":
                return 1 if x.get("kept") else 0
            elif sort_by == "timestamp_utc":
                ts = _parse_iso_timestamp(
                    x.get("checkpoint_updated_ts_utc") or x.get("timestamp_utc")
                )
                return ts or datetime.min.replace(tzinfo=timezone.utc)
            elif sort_by == "run_id":
                return x.get("run_id", "")
            elif sort_by == "worker_id":
                return x.get("worker_id", "")
            elif sort_by == "idea_title":
                return x.get("idea_title", "")
            elif sort_by == "job_id":
                return x.get("job_id", "")
            else:
                return ""

        reverse = sort_order == "desc"
        filtered.sort(key=_make_sort_value, reverse=reverse)

        total = len(filtered)
        start = (page - 1) * per_page
        end = start + per_page
        page_items = filtered[start:end]

        return _safe_jsonify(
            {
                "experiments": page_items,
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page if total > 0 else 0,
            }
        )

    @app.route("/api/workers")
    def api_workers() -> jsonify:
        worker_states = _load_worker_states(resolved_output_dir)
        leaderboard = _load_leaderboard(resolved_output_dir)
        workers_dir = resolved_output_dir / "workers"

        worker_metrics: dict[str, list[float]] = {}
        for entry in leaderboard:
            wid = entry.get("worker_id", "")
            metric = _to_finite_float(entry.get("unified_metric"))
            if wid and metric is not None:
                worker_metrics.setdefault(wid, []).append(metric)

        workers_data = []
        for worker_id, state in worker_states.items():
            metrics = worker_metrics.get(worker_id, [])
            heartbeat_ts = state.get("heartbeat_ts_utc", "")
            stage = state.get("worker_stage", "unknown")
            run_index = state.get("run_index", 0)

            expert_name = None
            if workers_dir.exists():
                worker_info_path = workers_dir / worker_id / "worker_info.json"
                if worker_info_path.exists():
                    worker_info = _load_json_state(worker_info_path)
                    expert = worker_info.get("expert")
                    if expert and expert.get("name"):
                        expert_name = expert["name"]

            workers_data.append(
                {
                    "worker_id": worker_id,
                    "role": expert_name
                    or ("expert" if "expert" in worker_id else "general"),
                    "heartbeat_ts_utc": heartbeat_ts,
                    "worker_stage": stage,
                    "run_index": run_index,
                    "active_run_id": state.get("active_run_id", ""),
                    "total_runs": len(metrics),
                    "best_metric": max(metrics) if metrics else None,
                    "avg_metric": sum(metrics) / len(metrics) if metrics else None,
                    "last_activity": state.get("last_cycle_ts_utc")
                    or state.get("worker_stage_ts_utc", ""),
                    "error": state.get("last_cycle_error_ts_utc") is not None,
                }
            )

        workers_data.sort(key=lambda x: x.get("worker_id", ""))

        return _safe_jsonify({"workers": workers_data, "total": len(workers_data)})

    @app.route("/api/worker/<worker_id>")
    def api_worker_detail(worker_id: str) -> jsonify:
        worker_path = resolved_output_dir / "workers" / worker_id
        if not worker_path.exists():
            return _safe_jsonify({"error": "Worker not found"}), 404

        state = _load_json_state(worker_path / "worker_state.json")
        info = _load_json_state(worker_path / "worker_info.json")
        memory_path = worker_path / "memory.md"
        memory = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""

        leaderboard = _load_leaderboard(resolved_output_dir)
        worker_finished_runs = [
            e
            for e in leaderboard
            if e.get("worker_id") == worker_id and e.get("status") == "finished"
        ]
        worker_finished_runs.sort(
            key=lambda x: _parse_iso_timestamp(x.get("timestamp_utc"))
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        worker_active_runs = []
        active_run_id = state.get("active_run_id", "")
        if active_run_id and state.get("worker_stage", "") != "idle":
            for run_dir in worker_path.iterdir():
                if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                    continue
                run_info_path = run_dir / "run_info.json"
                if not run_info_path.exists():
                    continue
                run_info = _load_json_state(run_info_path)
                ac = run_info.get("active_cycle", {})
                if ac.get("run_id") != active_run_id:
                    continue
                trial_state_path = run_dir / "run_state.json"
                trial_state = (
                    _load_json_state(trial_state_path)
                    if trial_state_path.exists()
                    else {}
                )
                idea_payload = run_info.get("idea_payload", {})
                worker_stage = state.get("worker_stage", "")
                worker_active_runs.append(
                    {
                        "run_id": active_run_id,
                        "worker_id": worker_id,
                        "idea_title": idea_payload.get("idea_title", "")
                        or ac.get("idea_title", ""),
                        "status": "running"
                        if worker_stage == "executing"
                        else "pending",
                        "runtime_seconds": trial_state.get("last_runtime_seconds"),
                        "unified_metric": None,
                        "timestamp_utc": trial_state.get(
                            "checkpoint_updated_ts_utc", ""
                        ),
                        "job_id": trial_state.get("job_id", ""),
                        "kept": False,
                    }
                )
                break

        run_dirs = []
        for d in sorted(worker_path.iterdir()):
            if d.is_dir() and d.name.startswith("run_"):
                run_index = int(d.name.split("_")[1])
                run_info = (
                    _load_json_state(d / "run_info.json")
                    if (d / "run_info.json").exists()
                    else {}
                )
                run_dirs.append(
                    {
                        "run_index": run_index,
                        "run_path": str(d),
                        "run_info": run_info,
                    }
                )

        return _safe_jsonify(
            {
                "worker_id": worker_id,
                "state": state,
                "info": info,
                "memory": memory,
                "runs": worker_active_runs + worker_finished_runs,
                "run_dirs": run_dirs,
            }
        )

    @app.route("/api/run/<run_id>")
    def api_run_detail(run_id: str) -> jsonify:
        leaderboard = _load_leaderboard(resolved_output_dir)
        entry = next((e for e in leaderboard if e.get("run_id") == run_id), None)

        workers_dir = resolved_output_dir / "workers"
        run_info_entry = None

        if not entry and workers_dir.exists():
            for worker_path in workers_dir.iterdir():
                if not worker_path.is_dir() or not worker_path.name.startswith(
                    "worker_"
                ):
                    continue
                worker_id = worker_path.name
                state = _load_json_state(worker_path / "worker_state.json")
                active_run_id = state.get("active_run_id", "")
                if active_run_id != run_id:
                    continue
                for run_dir in worker_path.iterdir():
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                        continue
                    run_info_path = run_dir / "run_info.json"
                    if not run_info_path.exists():
                        continue
                    run_info = _load_json_state(run_info_path)
                    ac = run_info.get("active_cycle", {})
                    if ac.get("run_id") != run_id:
                        continue
                    trial_state_path = run_dir / "run_state.json"
                    trial_state = (
                        _load_json_state(trial_state_path)
                        if trial_state_path.exists()
                        else {}
                    )
                    idea_payload = run_info.get("idea_payload", {})
                    worker_info = _load_json_state(worker_path / "worker_info.json")
                    expert = worker_info.get("expert", {})
                    worker_stage = state.get("worker_stage", "")
                    outline = (
                        idea_payload.get("idea_outline", "")
                        or idea_payload.get("rough_outline", "")
                        or ac.get("idea_outline", "")
                    )
                    run_info_entry = {
                        "run_id": run_id,
                        "worker_id": worker_id,
                        "worker_role": expert.get("name", "general")
                        if isinstance(expert, dict)
                        else "general",
                        "idea_title": idea_payload.get("idea_title", "")
                        or ac.get("idea_title", ""),
                        "idea_outline": outline,
                        "aider_plan_prompt": idea_payload.get("aider_plan_prompt", "")
                        or ac.get("aider_plan_prompt", ""),
                        "aider_impl_prompt": idea_payload.get("aider_impl_prompt", "")
                        or ac.get("aider_impl_prompt", ""),
                        "status": "running"
                        if worker_stage == "executing"
                        else "pending",
                        "runtime_seconds": trial_state.get("last_runtime_seconds"),
                        "job_id": trial_state.get("job_id", ""),
                        "worker_stage": worker_stage,
                        "timestamp_utc": trial_state.get(
                            "checkpoint_updated_ts_utc", ""
                        ),
                        "feature_branch": ac.get("feature_branch", ""),
                        "baseline_commit": ac.get("baseline_commit", ""),
                        "trial_commit": trial_state.get("trial_commit", ""),
                        "kept": False,
                    }
                    break
                if run_info_entry:
                    break

        if not entry and not run_info_entry:
            return _safe_jsonify({"error": "Run not found"}), 404

        if entry:
            entry = dict(entry)
        elif run_info_entry:
            entry = dict(run_info_entry)

        if workers_dir.exists():
            for worker_path in workers_dir.iterdir():
                if not worker_path.is_dir() or not worker_path.name.startswith(
                    "worker_"
                ):
                    continue
                for run_dir in worker_path.iterdir():
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                        continue
                    run_info_path = run_dir / "run_info.json"
                    if run_info_path.exists():
                        run_info = _load_json_state(run_info_path)
                        exp_outcome = run_info.get("experiment_outcome", {})
                        if exp_outcome.get("run_id") == run_id:
                            if exp_outcome.get("idea_outline"):
                                entry["idea_outline"] = exp_outcome["idea_outline"]
                            if exp_outcome.get("aider_plan_prompt"):
                                entry["aider_plan_prompt"] = exp_outcome[
                                    "aider_plan_prompt"
                                ]
                            if exp_outcome.get("aider_impl_prompt"):
                                entry["aider_impl_prompt"] = exp_outcome[
                                    "aider_impl_prompt"
                                ]
                            if exp_outcome.get("runtime_seconds") is not None:
                                entry["runtime_seconds"] = exp_outcome[
                                    "runtime_seconds"
                                ]
                            if exp_outcome.get("avg_gpu_util") is not None:
                                entry["avg_gpu_util"] = exp_outcome["avg_gpu_util"]
                            if exp_outcome.get("avg_gpu_memory") is not None:
                                entry["avg_gpu_memory"] = exp_outcome["avg_gpu_memory"]
                            break

        worker_id = entry.get("worker_id", "")
        if worker_id and workers_dir.exists():
            worker_info_path = workers_dir / worker_id / "worker_info.json"
            if worker_info_path.exists():
                worker_info = _load_json_state(worker_info_path)
                expert = worker_info.get("expert")
                if expert and expert.get("name"):
                    entry["expert_name"] = expert["name"]

        parent_history = _build_parent_history(leaderboard, run_id, workers_dir)

        return _safe_jsonify({"run": entry, "parent_history": parent_history})

    @app.route("/api/worker/<worker_id>/log")
    def api_worker_log(worker_id: str) -> jsonify:
        worker_path = resolved_output_dir / "workers" / worker_id
        if not worker_path.exists():
            return _safe_jsonify({"error": "Worker not found"}), 404

        log_path = worker_path / "output.log"
        if not log_path.exists():
            return _safe_jsonify({"error": "Log file not found"}), 404

        try:
            log_content = log_path.read_text(encoding="utf-8", errors="replace")
            max_chars = 500000
            if len(log_content) > max_chars:
                log_content = "... (truncated)\n" + log_content[-max_chars:]
            return _safe_jsonify({"log": log_content})
        except Exception as e:
            return _safe_jsonify({"error": f"Failed to read log: {str(e)}"}), 500

    @app.route("/api/global-log")
    def api_global_log() -> jsonify:
        limit = int(request.args.get("limit", 1000))
        offset = int(request.args.get("offset", 0))
        log_path = resolved_output_dir / "output.log"
        if not log_path.exists():
            return _safe_jsonify({"log": "", "total": 0, "has_more": False})
        try:
            all_lines = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            total = len(all_lines)
            start = max(0, total - offset - limit)
            end = total - offset if offset < total else total
            chunk = all_lines[start:end]
            reversed_chunk = list(reversed(chunk))
            has_more = offset + limit < total
            return _safe_jsonify(
                {
                    "log": "\n".join(reversed_chunk),
                    "total": total,
                    "has_more": has_more,
                    "offset": offset,
                }
            )
        except Exception as e:
            return _safe_jsonify({"error": f"Failed to read log: {str(e)}"}), 500

    return app


def main() -> int:
    """Run the Flask development server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Sciantist web UI server.")
    parser.add_argument(
        "--output-dir",
        default="./outputs/sciantist",
        help="Path to sciantist output directory",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    args = parser.parse_args()

    app = create_app(output_dir=args.output_dir)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
