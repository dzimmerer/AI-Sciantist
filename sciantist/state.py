"""State persistence and output path helpers."""

from __future__ import annotations

import fcntl
import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sciantist.config import ExperimentOutcome


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _resolve_output_dir(output_dir: str) -> Path:
    """Return the resolved output directory used for artifacts and state."""
    return Path(output_dir).expanduser().resolve()


def _load_state(path: Path) -> dict[str, Any]:
    """Load persistent loop state from disk."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    """Save persistent loop state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def _load_json_state(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk, returning empty state on missing or invalid files."""
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json_state(path: Path, state: dict[str, Any]) -> None:
    """Persist JSON state atomically to avoid partial writes on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(path)


def _load_trial_run_state(path: Path) -> dict[str, Any]:
    """Load a candidate-specific trial checkpoint state."""
    return _load_json_state(path)


def _save_trial_run_state(path: Path, state: dict[str, Any]) -> None:
    """Save candidate-specific trial checkpoint state with update timestamp."""
    next_state = dict(state)
    next_state["checkpoint_updated_ts_utc"] = _utc_now_iso()
    next_state.setdefault("schema_version", 1)
    _save_json_state(path, next_state)


def _build_trial_state_path(candidate_output_dir: Path) -> Path:
    """Return the checkpoint path used for one candidate run."""
    return candidate_output_dir / "run_state.json"


def _build_stage_plan_path(stage_dir: Path) -> Path:
    """Return the persisted stage plan path for resume-safe candidate reuse."""
    return stage_dir / "stage_plan.json"


def _load_stage_plan(path: Path) -> dict[str, Any]:
    """Load persisted stage plan state."""
    return _load_json_state(path)


def _save_stage_plan(path: Path, plan: dict[str, Any]) -> None:
    """Save persisted stage plan state."""
    _save_json_state(path, plan)


def _build_lock_path(output_root: Path, lock_file_name: str) -> Path:
    """Return a shared lock file path under the output directory."""
    return output_root / lock_file_name


@contextmanager
def _file_lock(lock_path: Path):
    """Acquire an exclusive advisory file lock for cross-process synchronization."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _build_leaderboard_path(output_root: Path, leaderboard_json_name: str) -> Path:
    """Return path for canonical leaderboard JSON state."""
    return output_root / leaderboard_json_name


def _load_leaderboard(path: Path) -> list[dict[str, Any]]:
    """Load leaderboard entries from JSON, defaulting to empty list."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _save_leaderboard(path: Path, entries: list[dict[str, Any]]) -> None:
    """Save canonical leaderboard JSON with atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(entries, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(path)


def _to_finite_metric(value: Any) -> float | None:
    """Return a finite float metric value, or None when value is invalid."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    metric = float(value)
    if not math.isfinite(metric):
        return None
    return metric


def _append_leaderboard_entry(
    leaderboard_path: Path,
    lock_path: Path,
    entry: dict[str, Any],
) -> bool:
    """Append one leaderboard entry idempotently by run_id under lock."""
    run_id = str(entry.get("run_id", "")).strip()
    if not run_id:
        return False

    with _file_lock(lock_path):
        entries = _load_leaderboard(leaderboard_path)
        for existing in entries:
            if str(existing.get("run_id", "")).strip() == run_id:
                return False
        entries.append(entry)

        # sort entries by unified metric and timestamp for easier human consumption (newest first for ties)
        def _sort_key(item: dict[str, Any]) -> tuple[float, str]:
            metric = _to_finite_metric(item.get("unified_metric"))
            return (
                metric if metric is not None else float("-inf"),
                str(item.get("timestamp_utc", "")),
            )

        entries.sort(
            key=_sort_key,
            reverse=True,  # higher unified_metric is better, and newer timestamps first for ties
        )

        _save_leaderboard(leaderboard_path, entries)
    return True


def _select_best_entry(
    entries: list[dict[str, Any]],
    higher_is_better: bool,
) -> dict[str, Any] | None:
    """Select best leaderboard entry by unified metric and status."""
    finished_with_metric: list[tuple[dict[str, Any], float]] = []
    for item in entries:
        if str(item.get("status", "")).strip() != "finished":
            continue
        metric_value = _to_finite_metric(item.get("unified_metric"))
        if metric_value is None:
            continue
        finished_with_metric.append((item, metric_value))

    if not finished_with_metric:
        return None

    chooser = max if higher_is_better else min
    best_item, _ = chooser(finished_with_metric, key=lambda pair: pair[1])
    return best_item


def _get_best_leaderboard_entry(
    leaderboard_path: Path,
    lock_path: Path,
    higher_is_better: bool,
) -> dict[str, Any] | None:
    """Return the current best leaderboard entry under lock."""
    with _file_lock(lock_path):
        entries = _load_leaderboard(leaderboard_path)
    return _select_best_entry(entries, higher_is_better)


def _refresh_leaderboard_best_flags(
    leaderboard_path: Path,
    lock_path: Path,
    metric_higher_is_better: bool,
) -> dict[str, Any] | None:
    """Recompute and persist currently_in_best_path flags under lock."""
    with _file_lock(lock_path):
        entries = _load_leaderboard(leaderboard_path)
        best = None
        for item in entries:
            if str(item.get("status", "")) != "finished":
                continue
            metric = _to_finite_metric(item.get("unified_metric"))
            if metric is None:
                continue
            if best is None:
                best = item
                continue
            best_metric = _to_finite_metric(best.get("unified_metric"))
            if best_metric is None:
                best = item
                continue
            if metric_higher_is_better and metric > best_metric:
                best = item
            if (not metric_higher_is_better) and metric < best_metric:
                best = item

        by_run_id: dict[str, dict[str, Any]] = {}
        for item in entries:
            run_id = str(item.get("run_id", "")).strip()
            if run_id:
                by_run_id[run_id] = item

        best_run_id = str(best.get("run_id", "")) if isinstance(best, dict) else ""
        best_path_ids: set[str] = set()
        if best_run_id:
            stack = [best_run_id]
            while stack:
                current = stack.pop()
                if current in best_path_ids:
                    continue
                best_path_ids.add(current)
                node = by_run_id.get(current)
                if node is None:
                    continue
                parents = node.get("parent_entry_ids")
                if isinstance(parents, list):
                    for parent in parents:
                        parent_id = str(parent).strip()
                        if parent_id:
                            stack.append(parent_id)

        changed = False
        for item in entries:
            run_id = str(item.get("run_id", "")).strip()
            next_flag = bool(run_id and run_id in best_path_ids)
            if bool(item.get("currently_in_best_path", False)) != next_flag:
                item["currently_in_best_path"] = next_flag
                changed = True

        if changed:
            _save_leaderboard(leaderboard_path, entries)
        return best


def _build_worker_state_path(output_root: Path, worker_id: str) -> Path:
    """Return persistent worker runtime-state path."""
    return output_root / "workers" / worker_id / "worker_state.json"


def _load_worker_runtime_state(path: Path) -> dict[str, Any]:
    """Load worker runtime checkpoint state."""
    return _load_json_state(path)


def _save_worker_runtime_state(path: Path, state: dict[str, Any]) -> None:
    """Save worker runtime checkpoint state with update timestamp."""
    next_state = dict(state)
    next_state["checkpoint_updated_ts_utc"] = _utc_now_iso()
    next_state.setdefault("schema_version", 1)
    _save_json_state(path, next_state)


def _load_worker_info(path: Path) -> dict[str, Any]:
    """Load worker info payload from disk, falling back to empty object."""
    return _load_json_state(path)


def _save_worker_info(path: Path, payload: dict[str, Any]) -> None:
    """Persist worker info payload atomically."""
    _save_json_state(path, payload)


def _expert_payload_from_spec(expert_spec: tuple[str, str] | None) -> dict[str, str] | None:
    """Build normalized expert payload for persisted JSON artifacts."""
    if expert_spec is None:
        return None
    return {
        "name": expert_spec[0],
        "description": expert_spec[1],
    }


def _update_worker_info(
    worker_info_path: Path,
    *,
    expert_spec: tuple[str, str] | None,
    current_run_index: int,
    run_id: str | None = None,
) -> None:
    """Update worker_info.json with assignment metadata and run history."""
    info = _load_worker_info(worker_info_path)
    run_ids_raw = info.get("previous_run_ids")
    run_ids = [
        str(item).strip() for item in (run_ids_raw if isinstance(run_ids_raw, list) else []) if str(item).strip()
    ]
    normalized_run_id = str(run_id).strip() if isinstance(run_id, str) else ""
    if normalized_run_id and normalized_run_id not in run_ids:
        run_ids.append(normalized_run_id)

    _save_worker_info(
        worker_info_path,
        {
            "expert": _expert_payload_from_spec(expert_spec),
            "current_run_index": int(current_run_index),
            "previous_run_ids": run_ids,
            "updated_ts_utc": _utc_now_iso(),
        },
    )


def _update_run_info(run_info_path: Path, patch: dict[str, Any]) -> None:
    """Merge and persist run_info.json atomically for one worker run."""
    existing = _load_worker_info(run_info_path)
    next_payload = dict(existing)
    next_payload.update(patch)
    next_payload["updated_ts_utc"] = _utc_now_iso()
    next_payload.setdefault("schema_version", 1)
    _save_worker_info(run_info_path, next_payload)


def _leaderboard_run_id_exists(leaderboard_path: Path, run_id: str) -> bool:
    """Return True when leaderboard already contains the specified run id."""
    target_run_id = str(run_id).strip()
    if not target_run_id:
        return False
    for entry in _load_leaderboard(leaderboard_path):
        if str(entry.get("run_id", "")).strip() == target_run_id:
            return True
    return False


def _merged_validation_entry_exists(leaderboard_path: Path, merge_run_id: str) -> bool:
    """Return True when leaderboard already contains merged-validation row for merge run."""
    for entry in _load_leaderboard(leaderboard_path):
        if str(entry.get("merged_validation_of", "")).strip() == merge_run_id:
            return True
    return False


def _serialize_experiment_outcome(outcome: ExperimentOutcome) -> dict[str, Any]:
    """Serialize experiment outcome for worker runtime checkpoint state."""
    return dict(outcome.__dict__)


def _deserialize_experiment_outcome(raw: Any) -> ExperimentOutcome | None:
    """Deserialize checkpointed outcome payload into ExperimentOutcome."""
    if not isinstance(raw, dict):
        return None
    try:
        return ExperimentOutcome(**raw)
    except TypeError:
        return None
