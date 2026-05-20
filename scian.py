"""Autonomous AI scientist loop for iterative codebase experiments.

This script continuously:
1. Generates a websearch-grounded experiment idea.
2. Uses aider to plan and implement code changes.
3. Commits the changes on a dedicated branch.
4. Submits and monitors a 2-hour SLURM experiment.
5. Pulls W&B metrics (run id == job id), computes a unified metric,
   and keeps or reverts the commit based on improvement.
6. Logs outcomes to experiments markdown and TSV ledgers.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from sciantist.aider_ops import (
    _resolve_merge_conflicts_with_aider,
    _run_aider_plan_and_impl,
    _try_fix_crash_with_aider,
)
from sciantist.cluster_ops import (
    _runtime_to_seconds,
    configure_active_cluster,
    get_active_cluster_profile,
    get_cluster_job_runtime,
    get_cluster_job_status,
    load_remote_log,
    submit_cluster_job,
)
from sciantist.config import (
    ExperimentOutcome,
    LoopConfig,
    StageOutcome,
    _load_repo_config_yaml,
)
from sciantist.ideation import (
    _build_file_summaries_for_prompt,
    _build_ideation_prompt,
    _build_improvement_brief_with_websearch,
    _build_stage_improvement_ideas_with_websearch,
    _build_tried_ideas_summary,
    _build_websearch_idea_prompt,
    _extend_ideation_prompt_with_expert,
    _extract_idea_payload_list,
    _format_websearch_ideas_for_ideation_context,
    _initialize_file_summary_cache_if_needed,
    _iter_candidate_files,
    _load_expert_specs,
    _normalize_idea_payload,
    _query_ideation_json,
    _query_stage_ideation_payloads,
    _read_codebase_glimpse,
    _read_text_if_exists,
    _slugify,
    _tail_text,
    _update_stage_memory_with_llm,
)
from sciantist.logging_utils import configure_logging
from sciantist.metrics import (
    _metric_is_better,
    _metric_matches_or_better,
    calc_unified_metric,
    extract_weighted_metric_inputs,
)
from sciantist.paths import (  # pyright: ignore[reportMissingImports]
    _build_leaderboard_lock_path,
    _build_memory_lock_path,
    _build_repo_lock_path,
    _build_worker_memory_path,
    _build_worker_output_dir,
    _build_worker_run_id,
    _build_worker_worktree_path,
)
from sciantist.repo_ops import (
    _attempt_merge_on_base_branch,
    _candidate_worktree_path,
    _checkout_feature_branch,
    _cleanup_candidate_worktree,
    _commit_if_dirty,
    _commit_tree_id,
    _create_candidate_worktree,
    _ensure_repo_ready,
    _extract_diff_patch,
    _extract_diff_summary,
    _git,
    _git_try,
    _make_feature_branch_name,
    _merge_feature_into_base,
    _push_to_origin,
    _resolve_train_command_for_repo,
)
from sciantist.reporting import (
    _append_experiments_md,
    _append_experiments_tsv,
    _format_runtime_dd_hh_mm_ss,
    _is_outcome_already_logged,
    _summarize_run_with_llm,
    _write_stage_candidate_summary_markdown,
    _write_stage_summary_markdown,
)
from sciantist.state import (
    _append_leaderboard_entry,
    _build_leaderboard_path,
    _build_stage_plan_path,
    _build_trial_state_path,
    _build_worker_state_path,
    _deserialize_experiment_outcome,
    _expert_payload_from_spec,
    _file_lock,
    _get_best_leaderboard_entry,
    _leaderboard_run_id_exists,
    _load_leaderboard,
    _load_stage_plan,
    _load_state,
    _load_trial_run_state,
    _load_worker_runtime_state,
    _merged_validation_entry_exists,
    _refresh_leaderboard_best_flags,
    _resolve_output_dir,
    _save_leaderboard,
    _save_stage_plan,
    _save_state,
    _save_trial_run_state,
    _save_worker_runtime_state,
    _serialize_experiment_outcome,
    _update_run_info,
    _update_worker_info,
    _utc_now_iso,
)
from sciantist.wandb_ops import create_tracking_backend


def _vprint(config: LoopConfig, message: str) -> None:
    """Print verbose log lines when verbose mode is enabled."""
    if config.verbose:
        logger.debug(message)


def _update_trial_polling_state(
    trial_state_path: Path,
    status: str,
    runtime_seconds: int | None,
) -> None:
    """Update in-flight polling details for checkpoint-based resume."""
    run_state = _load_trial_run_state(trial_state_path)
    run_state["polling_in_progress"] = True
    run_state["last_status"] = status
    run_state["last_runtime_seconds"] = runtime_seconds
    run_state["last_poll_ts_utc"] = _utc_now_iso()
    _save_trial_run_state(trial_state_path, run_state)


def _poll_job_until_terminal(
    job_id: str,
    runtime_budget_seconds: int,
    poll_seconds: int,
    config: LoopConfig,
    poll_checkpoint_callback: Callable[[str, int | None], None] | None = None,
    poll_context: str | None = None,
) -> tuple[str, int | None]:
    """Poll SLURM status/runtime until terminal or timeout guard triggers."""
    start = time.time()
    status = "unknown"
    runtime_seconds: int | None = None
    candidate_label = poll_context if poll_context else "unknown"
    while True:
        status = get_cluster_job_status(job_id)
        runtime_seconds = get_cluster_job_runtime(job_id)
        runtime_formatted = _format_runtime_dd_hh_mm_ss(runtime_seconds)
        logger.info(
            f"candidate={candidate_label} job={job_id} status={status} "
            f"runtime={runtime_seconds}s ({runtime_formatted})"
        )
        if poll_checkpoint_callback is not None:
            poll_checkpoint_callback(status, runtime_seconds)

        if status in {"finished", "crashed"}:
            logger.info(
                f"job={job_id} reached terminal status={status} with runtime={runtime_seconds}s. Runtime budget is {runtime_budget_seconds}s. IF statement: {runtime_seconds} is not None and {runtime_seconds} >= {runtime_budget_seconds - config.stop_before_secs}"
            )
            if (
                runtime_seconds is not None and runtime_seconds >= runtime_budget_seconds - config.stop_before_secs
            ):  # allow some buffer for post-processing
                logger.info(f"job={job_id} completed with status={status} within runtime budget.")
                status = "finished"

            logger.info(
                f"Exiting polling loop for job={job_id} with terminal status={status} and runtime={runtime_seconds}s."
            )
            return status, runtime_seconds

        wall_clock = int(time.time() - start)
        if wall_clock > runtime_budget_seconds + 6000:
            _vprint(
                config,
                (
                    f"Wall-clock guard tripped for job={job_id}. "
                    f"wall_clock={wall_clock}s budget={runtime_budget_seconds}s"
                ),
            )
            return "crashed", runtime_seconds

        time.sleep(max(1, poll_seconds))


def _run_on_cluster(
    config: LoopConfig,
    files_to_edit: list[str],
    idea_title: str,
    skip_aider: bool,
    trial_state_path: Path,
    wandb_output_dir: Path,
    expected_trial_commit: str,
    heartbeat_callback: Callable[[], None] | None = None,
) -> tuple[
    str,
    int | None,
    str,
    str,
    str,
    str,
    float | None,
    float | None,
    float | None,
    dict[str, list[float]],
]:
    """Execute/resume cluster run, retry crash fixes, and collect logs/metrics."""

    def _is_invalid_unified_metric(value: float | None) -> bool:
        """Treat None/NaN/inf as invalid unified metric values."""
        if value is None:
            return True
        return math.isnan(value) or math.isinf(value)

    status = "dry-run"
    runtime_seconds: int | None = None
    unified_metric: float | None = None
    avg_gpu_util: float | None = None
    avg_gpu_memory: float | None = None
    metric_histories: dict[str, list[float]] = {}
    info_log = ""
    err_log = ""
    wandb_log = ""
    job_id = "dry-run"

    runtime_budget_seconds = _runtime_to_seconds(config.runtime)
    run_state = _load_trial_run_state(trial_state_path)
    poll_context = str(run_state.get("candidate_label", "")).strip() if run_state else ""
    run_state.update(
        {
            "runtime_budget_seconds": runtime_budget_seconds,
            "poll_seconds": config.poll_seconds,
            "cluster_target": config.cluster_target,
            "train_command": config.train_command,
        }
    )
    _save_trial_run_state(trial_state_path, run_state)

    def _poll_checkpoint(status_value: str, runtime_value: int | None) -> None:
        _update_trial_polling_state(trial_state_path, status_value, runtime_value)
        if heartbeat_callback is not None:
            heartbeat_callback()

    def _retry_crash_with_aider(
        *,
        job_id_value: str,
        status_value: str,
        runtime_value: int | None,
        reason: str,
    ) -> tuple[str, str, int | None]:
        """Retry crashed runs by asking aider for fixes and resubmitting."""
        if status_value != "crashed" or skip_aider:
            return status_value, job_id_value, runtime_value

        for attempt in range(1, config.max_fix_attempts + 1):
            try:
                latest_err_log = load_remote_log(job_id_value, log="ERROR")
            except RuntimeError:
                latest_err_log = ""

            try:
                latest_info_log = load_remote_log(job_id_value, log="INFO")
            except RuntimeError:
                latest_info_log = ""

            fixed_commit = _try_fix_crash_with_aider(
                config=config,
                files_to_edit=files_to_edit,
                idea_title=idea_title,
                err_log=latest_err_log or latest_info_log,
                attempt=attempt,
            )
            if not fixed_commit:
                _vprint(
                    config,
                    f"Crash fix attempt {attempt} produced no commit ({reason}); stopping retries.",
                )
                break

            logger.info(f"re-submit after fix attempt {attempt}, commit={fixed_commit}, reason={reason}")
            submit_command = _resolve_train_command_for_repo(
                config.train_command,
                config.original_target_repo,
                config.target_repo,
            )
            job_id_int = submit_cluster_job(
                command=submit_command,
                runtime=config.runtime,
                extra_args=config.cluster_extra_args,
                cluster_target=config.cluster_target,
                runner_script=config.cluster_runner_script,
            )
            job_id_value = str(job_id_int)
            run_state_local = _load_trial_run_state(trial_state_path)
            run_state_local["trial_commit"] = fixed_commit
            run_state_local["job_id"] = job_id_value
            run_state_local["polling_in_progress"] = True
            run_state_local["terminal_status"] = None
            run_state_local["last_fix_attempt"] = attempt
            run_state_local["fix_reason"] = reason
            _save_trial_run_state(trial_state_path, run_state_local)
            status_value, runtime_value = _poll_job_until_terminal(
                job_id=job_id_value,
                runtime_budget_seconds=runtime_budget_seconds,
                poll_seconds=config.poll_seconds,
                config=config,
                poll_checkpoint_callback=_poll_checkpoint,
                poll_context=poll_context,
            )
            run_state_local = _load_trial_run_state(trial_state_path)
            run_state_local["polling_in_progress"] = False
            run_state_local["terminal_status"] = status_value
            run_state_local["last_status"] = status_value
            run_state_local["last_runtime_seconds"] = runtime_value
            _save_trial_run_state(trial_state_path, run_state_local)
            if status_value == "finished":
                break

        return status_value, job_id_value, runtime_value

    resume_job_id = run_state.get("job_id") if isinstance(run_state.get("job_id"), str) else None
    resume_terminal_status = (
        run_state.get("terminal_status") if isinstance(run_state.get("terminal_status"), str) else None
    )
    if (
        resume_job_id
        and not bool(run_state.get("outcome_persisted", False))
        and run_state.get("trial_commit") == expected_trial_commit
    ):
        job_id = resume_job_id
        _vprint(config, f"Resuming existing candidate job polling: job={job_id} state={trial_state_path}")
        if resume_terminal_status in {"finished", "crashed"}:
            status = resume_terminal_status
            saved_runtime = run_state.get("last_runtime_seconds")
            runtime_seconds = saved_runtime if isinstance(saved_runtime, int) else None
        else:
            status, runtime_seconds = _poll_job_until_terminal(
                job_id=job_id,
                runtime_budget_seconds=runtime_budget_seconds,
                poll_seconds=config.poll_seconds,
                config=config,
                poll_checkpoint_callback=_poll_checkpoint,
                poll_context=poll_context,
            )
    else:
        submit_command = _resolve_train_command_for_repo(
            config.train_command,
            config.original_target_repo,
            config.target_repo,
        )
        _vprint(
            config,
            (
                f"Submitting cluster job command={submit_command} runtime={config.runtime} "
                f"extra_args={config.cluster_extra_args!r} target={config.cluster_target}"
            ),
        )
        job_id_int = submit_cluster_job(
            command=submit_command,
            runtime=config.runtime,
            extra_args=config.cluster_extra_args,
            cluster_target=config.cluster_target,
            runner_script=config.cluster_runner_script,
        )
        job_id = str(job_id_int)
        _vprint(config, f"Submitted job id: {job_id}")
        run_state = _load_trial_run_state(trial_state_path)
        run_state["job_id"] = job_id
        run_state["polling_in_progress"] = True
        run_state["terminal_status"] = None
        _save_trial_run_state(trial_state_path, run_state)
        status, runtime_seconds = _poll_job_until_terminal(
            job_id=job_id,
            runtime_budget_seconds=runtime_budget_seconds,
            poll_seconds=config.poll_seconds,
            config=config,
            poll_checkpoint_callback=_poll_checkpoint,
            poll_context=poll_context,
        )

    run_state = _load_trial_run_state(trial_state_path)
    run_state["job_id"] = job_id
    run_state["polling_in_progress"] = False
    run_state["terminal_status"] = status
    run_state["last_status"] = status
    run_state["last_runtime_seconds"] = runtime_seconds
    _save_trial_run_state(trial_state_path, run_state)

    status, job_id, runtime_seconds = _retry_crash_with_aider(
        job_id_value=job_id,
        status_value=status,
        runtime_value=runtime_seconds,
        reason="cluster-status-crashed",
    )

    try:
        info_log = load_remote_log(job_id, log="INFO")
    except RuntimeError:
        info_log = info_log or ""
    try:
        err_log = load_remote_log(job_id, log="ERROR")
    except RuntimeError:
        err_log = err_log or ""
    run_state = _load_trial_run_state(trial_state_path)
    run_state["logs_loaded"] = True
    _save_trial_run_state(trial_state_path, run_state)

    metric_recovery_attempted = False
    tracking_backend = create_tracking_backend(config.tracking_backend)
    should_sync_wandb = get_active_cluster_profile().wandb_sync
    while status == "finished" and config.wandb_project:
        _vprint(config, f"Fetching W&B snapshot for project={config.wandb_project} run/job={job_id}")
        if should_sync_wandb:
            tracking_backend.sync_from_cluster(log_fn=lambda message: _vprint(config, message))
        else:
            _vprint(config, "Skipping W&B sync_from_cluster due to active cluster profile setting wandb_sync=false")
        try:
            wandb_payload = tracking_backend.fetch_run_snapshot_by_job_id(
                project_path=config.wandb_project,
                job_id=job_id,
                output_dir=wandb_output_dir,
                metric_names=list(config.metric_weights.keys()),
                max_wandb_retries=config.max_wandb_retries,
            )
        except RuntimeError as error:
            if "No run named" in str(error):
                logger.warning(
                    "Skipping W&B snapshot for job_id=%s in project=%s: %s",
                    job_id,
                    config.wandb_project,
                    error,
                )
                _vprint(
                    config,
                    (
                        f"No W&B run found for job_id={job_id} in project={config.wandb_project}; "
                        "continuing without W&B metrics."
                    ),
                )
                run_state = _load_trial_run_state(trial_state_path)
                status = "crashed"
                run_state["terminal_status"] = status
                run_state["last_status"] = status
                run_state["wandb_fetched"] = False
                run_state["wandb_missing_run"] = True
                _save_trial_run_state(trial_state_path, run_state)
                if not metric_recovery_attempted:
                    metric_recovery_attempted = True
                    status, job_id, runtime_seconds = _retry_crash_with_aider(
                        job_id_value=job_id,
                        status_value=status,
                        runtime_value=runtime_seconds,
                        reason="wandb-run-missing",
                    )
                    continue
            else:
                raise
        else:
            wandb_log = tracking_backend.load_output_log_excerpt(wandb_payload, max_chars=8000)
            avg_gpu_util, avg_gpu_memory = tracking_backend.fetch_avg_gpu_metrics_from_events(
                project_path=config.wandb_project,
                job_id=job_id,
            )
            metrics, metric_histories = extract_weighted_metric_inputs(wandb_payload, config.metric_weights)
            _vprint(config, f"Metric keys available: {sorted(metrics.keys())[:80]}")
            _vprint(config, f"Tracked metric history keys: {sorted(metric_histories.keys())[:80]}")
            unified_metric = calc_unified_metric(metrics, config.metric_weights)
            _vprint(config, f"Unified metric computed: {unified_metric}")
            _vprint(config, f"Average GPU util={avg_gpu_util}, average GPU memory={avg_gpu_memory}")

            if _is_invalid_unified_metric(unified_metric):
                logger.warning(
                    (
                        "Unified metric invalid for job_id=%s (value=%s). "
                        "Treating run as crashed and attempting aider-based recovery."
                    ),
                    job_id,
                    unified_metric,
                )
                status = "crashed"
                run_state = _load_trial_run_state(trial_state_path)
                run_state["terminal_status"] = status
                run_state["last_status"] = status
                run_state["unified_metric_invalid"] = True
                run_state["unified_metric_value"] = str(unified_metric)
                _save_trial_run_state(trial_state_path, run_state)
                if not skip_aider and not metric_recovery_attempted:
                    metric_recovery_attempted = True
                    status, job_id, runtime_seconds = _retry_crash_with_aider(
                        job_id_value=job_id,
                        status_value=status,
                        runtime_value=runtime_seconds,
                        reason="invalid-unified-metric",
                    )
                    continue

            run_state = _load_trial_run_state(trial_state_path)
            run_state["wandb_fetched"] = True
            run_state["wandb_missing_run"] = False
            _save_trial_run_state(trial_state_path, run_state)
        break

    return (
        status,
        runtime_seconds,
        job_id,
        info_log,
        err_log,
        wandb_log,
        unified_metric,
        avg_gpu_util,
        avg_gpu_memory,
        metric_histories,
    )


def _execute_plan(
    config: LoopConfig,
    files_to_edit: list[str],
    baseline_commit: str,
    feature_branch: str,
    idea_title: str,
    idea_branch_name: str,
    idea_outline: str,
    aider_plan_prompt: str,
    aider_impl_prompt: str,
    baseline_metric_value: float | None,
    wandb_output_dir: Path,
    candidate_output_dir: Path,
    stage_index: int,
    candidate_label: str,
    skip_aider: bool = False,
    heartbeat_callback: Callable[[], None] | None = None,
) -> ExperimentOutcome:
    """Run one candidate trial from an already prepared branch checkout."""
    trial_state_path = _build_trial_state_path(candidate_output_dir)
    existing_state = _load_trial_run_state(trial_state_path)
    state_is_compatible = (
        # existing_state.get("feature_branch") == feature_branch
        existing_state.get("baseline_commit") == baseline_commit
        and existing_state.get("stage_index") == stage_index
        and existing_state.get("candidate_label") == candidate_label
        and idea_branch_name
        in existing_state.get("feature_branch", "")  # allow some flexibility in matching branch name
    )
    run_state: dict[str, Any] = existing_state if state_is_compatible else {}
    if run_state:
        _vprint(config, f"Loaded checkpoint state: {trial_state_path}")

    run_state.update(
        {
            "schema_version": 1,
            "stage_index": stage_index,
            "candidate_label": candidate_label,
            "feature_branch": feature_branch,
            "baseline_commit": baseline_commit,
            "idea_title": idea_title,
            "trial_repo": str(config.target_repo),
            "candidate_output_dir": str(candidate_output_dir),
        }
    )

    resume_trial_commit = run_state.get("trial_commit") if isinstance(run_state.get("trial_commit"), str) else None
    if resume_trial_commit and not config.dry_run and not skip_aider:
        _vprint(config, f"Resuming candidate code from prior trial_commit={resume_trial_commit}")
        _git(config.target_repo, "checkout", feature_branch)
        _git(config.target_repo, "reset", "--hard", resume_trial_commit)
        trial_commit = resume_trial_commit
    elif config.dry_run or skip_aider:
        trial_commit = _git(config.target_repo, "rev-parse", "HEAD")
    else:
        pre_impl_commit = _git(config.target_repo, "rev-parse", "HEAD")
        _run_aider_plan_and_impl(
            config,
            files_to_edit,
            aider_plan_prompt,
            aider_impl_prompt,
            commit_message=idea_title,
        )
        post_impl_commit = _git(config.target_repo, "rev-parse", "HEAD")
        if post_impl_commit != pre_impl_commit:
            trial_commit = post_impl_commit
            _vprint(config, f"Trial commit created by aider: {trial_commit} with message: {idea_title}")
        else:
            fallback_commit = _commit_if_dirty(config.target_repo, idea_title)
            trial_commit = fallback_commit if fallback_commit else post_impl_commit
            _vprint(
                config,
                (
                    f"Aider did not create commit; fallback commit result={fallback_commit}. "
                    f"Using trial_commit={trial_commit}"
                ),
            )

    run_state["trial_commit"] = trial_commit
    run_state["outcome_persisted"] = bool(run_state.get("outcome_persisted", False))
    _save_trial_run_state(trial_state_path, run_state)

    (
        status,
        runtime_seconds,
        job_id,
        info_log,
        err_log,
        wandb_log,
        unified_metric,
        avg_gpu_util,
        avg_gpu_memory,
        metric_histories,
    ) = _run_on_cluster(
        config=config,
        files_to_edit=files_to_edit,
        idea_title=idea_title,
        skip_aider=skip_aider,
        trial_state_path=trial_state_path,
        wandb_output_dir=wandb_output_dir,
        expected_trial_commit=trial_commit,
        heartbeat_callback=heartbeat_callback,
    )

    metric_delta: float | None = None
    if unified_metric is not None and baseline_metric_value is not None:
        metric_delta = unified_metric - baseline_metric_value
    elif unified_metric is not None:
        metric_delta = unified_metric

    diff_summary = _extract_diff_summary(config.target_repo, baseline_commit, trial_commit)
    if skip_aider:
        decision_summary = "Stage baseline run completed."
    elif config.dry_run:
        decision_summary = "Dry run completed; merge decision deferred to stage controller."
    elif status == "finished" and unified_metric is not None:
        decision_summary = "Candidate run finished with metric; merge decision deferred to stage controller."
    else:
        decision_summary = "Candidate run failed or missing metric; merge decision deferred to stage controller."
    final_summary = f"{decision_summary} {diff_summary}"

    run_state = _load_trial_run_state(trial_state_path)
    run_state["terminal_status"] = status
    run_state["last_status"] = status
    run_state["last_runtime_seconds"] = runtime_seconds
    run_state["outcome_persisted"] = False
    run_state["finalized_ts_utc"] = _utc_now_iso()
    _save_trial_run_state(trial_state_path, run_state)

    return ExperimentOutcome(
        timestamp_utc=_utc_now_iso(),
        idea_title=idea_title,
        idea_branch_name=idea_branch_name,
        feature_branch=feature_branch,
        idea_outline=idea_outline,
        aider_plan_prompt=aider_plan_prompt,
        aider_impl_prompt=aider_impl_prompt,
        baseline_commit=baseline_commit,
        trial_commit=trial_commit,
        job_id=job_id,
        wandb_project=config.wandb_project,
        status=status,
        runtime_seconds=runtime_seconds,
        unified_metric=unified_metric,
        metric_histories=metric_histories,
        baseline_metric=baseline_metric_value,
        metric_delta=metric_delta,
        avg_gpu_util=avg_gpu_util,
        avg_gpu_memory=avg_gpu_memory,
        kept=False,
        summary=final_summary,
        info_log_excerpt=_tail_text(info_log, 4000),
        wandb_log_excerpt=_tail_text(wandb_log, 4000),
        err_log_excerpt=_tail_text(err_log, 4000),
    )


def _touch_worker_heartbeat(
    worker_state_path: Path,
    worker_state: dict[str, Any],
    *,
    phase: str,
) -> None:
    """Persist worker heartbeat fields for supervisor stale detection."""
    worker_state["heartbeat_ts_utc"] = _utc_now_iso()
    worker_state["heartbeat_pid"] = os.getpid()
    worker_state["heartbeat_phase"] = phase
    _save_worker_runtime_state(worker_state_path, worker_state)


def _ensure_pending_merged_validation_entry(worker_state: dict[str, Any], item: dict[str, Any]) -> None:
    """Insert pending merged-validation item only when merge_run_id is not present."""
    pending = worker_state.get("pending_merged_validations")
    if not isinstance(pending, list):
        pending = []
    merge_run_id = str(item.get("merge_run_id", "")).strip()
    for existing in pending:
        if isinstance(existing, dict) and str(existing.get("merge_run_id", "")).strip() == merge_run_id:
            return
    pending.append(item)
    worker_state["pending_merged_validations"] = pending


def _append_outcome_to_shared_logs(
    config: LoopConfig,
    output_root: Path,
    lock_path: Path,
    leaderboard_path: Path,
    outcome: ExperimentOutcome,
    role: str,
    parent_entry_ids: list[str] | None = None,
    merged_validation_of: str = "",
) -> None:
    """Persist run output to experiments ledgers and leaderboard under lock."""
    experiments_md_path = output_root / config.experiments_md_name
    experiments_tsv_path = output_root / config.experiments_tsv_name

    with _file_lock(lock_path):
        if not _is_outcome_already_logged(experiments_tsv_path, outcome):
            _append_experiments_md(experiments_md_path, outcome)
            _append_experiments_tsv(experiments_tsv_path, outcome)

    entry = {
        "run_id": outcome.run_id,
        "worker_id": outcome.worker_id,
        "worker_role": role,
        "idea_title": outcome.idea_title,
        "idea_branch_name": outcome.idea_branch_name,
        "feature_branch": outcome.feature_branch,
        "baseline_commit": outcome.baseline_commit,
        "trial_commit": outcome.trial_commit,
        "status": outcome.status,
        "job_id": outcome.job_id,
        "unified_metric": outcome.unified_metric,
        "baseline_metric": outcome.baseline_metric,
        "metric_delta": outcome.metric_delta,
        "kept": outcome.kept,
        "summary": outcome.summary,
        "merged_validation_of": merged_validation_of,
        "parent_entry_ids": list(parent_entry_ids or []),
        "currently_in_best_path": False,
        "timestamp_utc": outcome.timestamp_utc,
    }
    _append_leaderboard_entry(leaderboard_path, lock_path, entry)
    _refresh_leaderboard_best_flags(leaderboard_path, lock_path, config.metric_higher_is_better)


def _resolve_current_best_baseline(
    config: LoopConfig,
    output_root: Path,
    lock_path: Path,
    leaderboard_path: Path,
) -> tuple[str, float | None]:
    """Resolve current best baseline commit+metric from leaderboard or state fallback."""
    best_entry = _get_best_leaderboard_entry(leaderboard_path, lock_path, config.metric_higher_is_better)
    return _resolve_current_best_baseline_with_best_entry(best_entry, config, output_root)


def _resolve_current_best_baseline_with_best_entry(
    best_entry: dict[str, Any] | None,
    config: LoopConfig,
    output_root: Path,
) -> tuple[str, float | None]:
    """Resolve current best baseline commit+metric from leaderboard or state fallback."""
    if best_entry:
        best_commit = str(best_entry.get("trial_commit", "")).strip()
        best_metric_raw = best_entry.get("unified_metric")
        best_metric = float(best_metric_raw) if isinstance(best_metric_raw, (float, int)) else None
        if best_commit:
            return best_commit, best_metric

    state_path = output_root / config.state_file_name
    state = _load_state(state_path)
    if isinstance(state.get("best_commit"), str) and state.get("best_commit"):
        metric = state.get("best_unified_metric")
        best_metric = float(metric) if isinstance(metric, (float, int)) else None
        return str(state["best_commit"]), best_metric

    _git(config.target_repo, "checkout", config.branch_name)
    return _git(config.target_repo, "rev-parse", "HEAD"), None


def _try_reuse_current_outcome_for_merged_validation(
    *,
    config: LoopConfig,
    output_root: Path,
    leaderboard_lock_path: Path,
    memory_lock_path: Path,
    leaderboard_path: Path,
    worker_id: str,
    merge_run_id: str,
    merged_commit: str,
    candidate_repo: Path,
    outcome: ExperimentOutcome,
) -> bool:
    """Reuse current run result as merged validation when merged tree is unchanged."""
    if _merged_validation_entry_exists(leaderboard_path, merge_run_id):
        return True

    merged_tree = _commit_tree_id(config.target_repo, merged_commit)
    current_tree = _commit_tree_id(candidate_repo, outcome.trial_commit)
    if not merged_tree or not current_tree or merged_tree != current_tree:
        return False

    reused = replace(outcome)
    reused.run_id = f"mergeval-{merge_run_id}"
    reused.worker_id = worker_id
    reused.worker_role = "merged_validation"
    reused.merged_validation_of = merge_run_id
    reused.summary = f"Reused worker run for merged validation (identical commit tree). {outcome.summary}"

    _append_outcome_to_shared_logs(
        config=config,
        output_root=output_root,
        lock_path=leaderboard_lock_path,
        leaderboard_path=leaderboard_path,
        outcome=reused,
        role="merged_validation",
        parent_entry_ids=[merge_run_id],
        merged_validation_of=merge_run_id,
    )
    with _file_lock(memory_lock_path):
        _update_stage_memory_with_llm(
            config=config,
            memory_path=output_root / "memory.md",
            stage_index=int(time.time()),
            stage_summary_text=f"{worker_id} reused current run as merged validation for {merge_run_id}",
            stage_outcomes=[reused],
        )
    logger.info(f"Reused current run as merged-validation outcome for run={merge_run_id}")
    return True


def _run_merged_validation_process(
    config: LoopConfig,
    worker_id: str,
    merge_run_id: str,
    merged_commit: str,
    baseline_metric: float | None,
) -> None:
    """Dedicated process for merged validation lifecycle (launch, monitor, metrics, leaderboard)."""
    output_root = _resolve_output_dir(config.output_dir)
    leaderboard_lock_path = _build_leaderboard_lock_path(output_root, config)
    memory_lock_path = _build_memory_lock_path(output_root)
    leaderboard_path = _build_leaderboard_path(output_root, config.leaderboard_json_name)
    run_index = int(time.time())
    run_id = f"mergeval-{merge_run_id}"
    worktree_path = output_root / "worktrees" / "merged_validation" / run_id
    candidate_output_dir = output_root / "merged_validation" / run_id
    feature_branch = _make_feature_branch_name("merged-validation", suffix=run_id[-10:])

    try:
        candidate_repo = _create_candidate_worktree(
            main_repo=config.target_repo,
            worktree_path=worktree_path,
            feature_branch=feature_branch,
            baseline_commit=merged_commit,
        )
        worker_config = replace(
            config,
            target_repo=str(candidate_repo),
            train_command=_resolve_train_command_for_repo(
                config.train_command,
                config.original_target_repo,
                str(candidate_repo),
            ),
        )
        files_to_edit = _iter_candidate_files(
            candidate_repo,
            worker_config.denylist_patterns,
            worker_config.aider_only_patterns,
            include_aider_only=False,
            allowed_suffixes=worker_config.allowed_file_suffixes,
        )
        outcome = _execute_plan(
            config=worker_config,
            files_to_edit=files_to_edit,
            baseline_commit=merged_commit,
            feature_branch=feature_branch,
            idea_title="merged_validation",
            idea_branch_name="merged_validation",
            idea_outline="Validation run for merged candidate.",
            aider_plan_prompt="",
            aider_impl_prompt="",
            baseline_metric_value=baseline_metric,
            wandb_output_dir=candidate_output_dir / "wandb_pull",
            candidate_output_dir=candidate_output_dir,
            stage_index=run_index,
            candidate_label=f"merged_validation_{worker_id}",
            skip_aider=True,
        )
        diff_patch = _extract_diff_patch(str(candidate_repo), outcome.baseline_commit, outcome.trial_commit)
        outcome.summary = _summarize_run_with_llm(
            config=config,
            outcome=outcome,
            diff_patch=diff_patch,
            decision_summary=outcome.summary,
        )
        outcome.run_id = run_id
        outcome.worker_id = worker_id
        outcome.worker_role = "merged_validation"
        outcome.merged_validation_of = merge_run_id
        _append_outcome_to_shared_logs(
            config=config,
            output_root=output_root,
            lock_path=leaderboard_lock_path,
            leaderboard_path=leaderboard_path,
            outcome=outcome,
            role="merged_validation",
            parent_entry_ids=[merge_run_id],
            merged_validation_of=merge_run_id,
        )
        with _file_lock(memory_lock_path):
            _update_stage_memory_with_llm(
                config=config,
                memory_path=output_root / "memory.md",
                stage_index=run_index,
                stage_summary_text=f"{worker_id} merged validation completed for {merge_run_id}",
                stage_outcomes=[outcome],
            )
    except Exception as error:
        logger.exception(f"Merged validation process failed for {merge_run_id}: {error}")
    finally:
        _cleanup_candidate_worktree(config.target_repo, worktree_path)


def _spawn_merged_validation_process(
    config: LoopConfig,
    worker_id: str,
    merge_run_id: str,
    merged_commit: str,
    baseline_metric: float | None,
) -> multiprocessing.Process:
    """Start dedicated process for merged validation lifecycle."""
    process = multiprocessing.Process(
        target=_run_merged_validation_process,
        args=(config, worker_id, merge_run_id, merged_commit, baseline_metric),
        name=f"sciantist-mergeval-{worker_id}",
        daemon=False,
    )
    process.start()
    return process


def _run_aider_merge_resolution_process(
    config: LoopConfig,
    worker_id: str,
    merge_run_id: str,
    feature_branch: str,
    baseline_metric: float | None,
) -> None:
    """Resolve merge conflicts with aider in an isolated worktree, then trigger validation."""
    output_root = _resolve_output_dir(config.output_dir)
    repo_lock_path = _build_repo_lock_path(output_root)
    run_id = f"mergefix-{merge_run_id}"
    worktree_path = output_root / "worktrees" / "merge_resolution" / run_id
    merge_branch = _make_feature_branch_name("merge-resolution", suffix=run_id[-10:])
    candidate_repo: Path | None = None

    try:
        with _file_lock(repo_lock_path):
            _git(config.target_repo, "checkout", config.branch_name)
            base_commit = _git(config.target_repo, "rev-parse", "HEAD")

        candidate_repo = _create_candidate_worktree(
            main_repo=config.target_repo,
            worktree_path=worktree_path,
            feature_branch=merge_branch,
            baseline_commit=base_commit,
        )
        worker_config = replace(config, target_repo=str(candidate_repo))

        ff_ok, ff_output = _git_try(str(candidate_repo), "merge", "--ff-only", feature_branch)
        noff_output = ""
        if not ff_ok:
            noff_ok, noff_output = _git_try(str(candidate_repo), "merge", "--no-ff", "--no-commit", feature_branch)
            if not noff_ok:
                conflicts_ok, conflicts_output = _git_try(
                    str(candidate_repo), "diff", "--name-only", "--diff-filter=U"
                )
                conflict_paths = (
                    [line.strip() for line in conflicts_output.splitlines() if line.strip()] if conflicts_ok else []
                )
                _resolve_merge_conflicts_with_aider(
                    worker_config,
                    candidate_repo,
                    conflict_paths,
                    feature_branch,
                    noff_output or ff_output,
                )

                unresolved_ok, unresolved_output = _git_try(
                    str(candidate_repo), "diff", "--name-only", "--diff-filter=U"
                )
                unresolved_paths = (
                    [line.strip() for line in unresolved_output.splitlines() if line.strip()] if unresolved_ok else []
                )
                if unresolved_paths:
                    raise RuntimeError(
                        f"Aider left unresolved merge files for {feature_branch}: {', '.join(unresolved_paths)}"
                    )

            committed = _commit_if_dirty(str(candidate_repo), f"merge resolved via aider: {feature_branch}")
            if committed is None:
                _git(
                    str(candidate_repo), "commit", "--allow-empty", "-m", f"merge resolved via aider: {feature_branch}"
                )

        merged_ok, merged_commit = _attempt_merge_on_base_branch(
            config=config,
            lock_path=repo_lock_path,
            feature_branch=merge_branch,
        )
        if not merged_ok:
            raise RuntimeError(f"Resolved merge branch could not be merged into base: {merge_branch}")

        push_ok, push_output = _push_to_origin(config.target_repo)
        if push_ok:
            logger.info(f"Pushed aider-resolved merge to origin: {push_output}")
        else:
            logger.warning(f"Failed to push aider-resolved merge to origin: {push_output}")

        merged_validation_proc = _spawn_merged_validation_process(
            config=config,
            worker_id=worker_id,
            merge_run_id=merge_run_id,
            merged_commit=merged_commit,
            baseline_metric=baseline_metric,
        )
        logger.info(
            f"Spawned merged-validation process pid={merged_validation_proc.pid} after aider merge resolution for run={merge_run_id}"
        )
    except Exception as error:
        logger.exception(f"Aider merge-resolution process failed for run={merge_run_id}: {error}")
    finally:
        if candidate_repo is not None:
            _cleanup_candidate_worktree(config.target_repo, worktree_path)

        with _file_lock(repo_lock_path):
            _git_try(config.target_repo, "branch", "-D", merge_branch)
            if not config.keep_failed_feature_branches:
                _git_try(config.target_repo, "branch", "-D", feature_branch)


def _spawn_aider_merge_resolution_process(
    config: LoopConfig,
    worker_id: str,
    merge_run_id: str,
    feature_branch: str,
    baseline_metric: float | None,
) -> multiprocessing.Process:
    """Start a dedicated process that resolves merge conflicts using aider."""
    merge_worker_id = f"{worker_id}_merge"
    process = multiprocessing.Process(
        target=_run_aider_merge_resolution_process,
        args=(config, merge_worker_id, merge_run_id, feature_branch, baseline_metric),
        name=f"sciantist-mergefix-{merge_worker_id}",
        daemon=False,
    )
    process.start()
    return process


def _run_async_worker_loop(
    config: LoopConfig,
    worker_index: int,
    worker_role: str,
    expert_spec: tuple[str, str] | None,
    run_once: bool,
) -> None:
    """Run autonomous worker loop with persisted stage-aware resume semantics."""
    output_root = _resolve_output_dir(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    leaderboard_lock_path = _build_leaderboard_lock_path(output_root, config)
    repo_lock_path = _build_repo_lock_path(output_root)
    memory_lock_path = _build_memory_lock_path(output_root)
    leaderboard_path = _build_leaderboard_path(output_root, config.leaderboard_json_name)
    worker_id = f"worker_{worker_index:02d}"
    worker_state_path = _build_worker_state_path(output_root, worker_id)
    worker_memory_path = _build_worker_memory_path(output_root, worker_id)
    worker_info_path = worker_memory_path.parent / "worker_info.json"
    worker_memory_path.parent.mkdir(parents=True, exist_ok=True)
    if not worker_memory_path.exists():
        worker_memory_path.write_text("# Experiment Memory\n\n", encoding="utf-8")
    worker_state = _load_worker_runtime_state(worker_state_path)
    run_index = int(worker_state.get("run_index", 0)) if isinstance(worker_state.get("run_index"), int) else 0

    _update_worker_info(
        worker_info_path,
        expert_spec=expert_spec,
        current_run_index=run_index,
    )

    _save_worker_runtime_state(worker_state_path, worker_state)

    while True:
        _touch_worker_heartbeat(worker_state_path, worker_state, phase="cycle-start")

        persisted_cycle = worker_state.get("active_cycle")
        active_cycle = persisted_cycle if isinstance(persisted_cycle, dict) else {}
        resumed_stage_raw = worker_state.get("worker_stage")
        resumed_stage = str(resumed_stage_raw).strip().lower() if isinstance(resumed_stage_raw, str) else ""

        resumed = False
        run_id = ""
        if active_cycle and resumed_stage in {"ideation", "executing", "evaluating"}:
            run_index_raw = active_cycle.get("run_index")
            run_id_raw = active_cycle.get("run_id")
            if isinstance(run_index_raw, int) and isinstance(run_id_raw, str) and run_id_raw.strip():
                run_index = run_index_raw
                run_id = run_id_raw.strip()
                resumed = True
                logger.info(f"Resuming {worker_id} run={run_id} from stage={resumed_stage}")

        if not resumed:
            run_index += 1
            run_id = _build_worker_run_id(worker_id, run_index)
            active_cycle = {
                "run_index": run_index,
                "run_id": run_id,
                "stage": "ideation",
            }
            resumed_stage = "ideation"
            worker_state["active_cycle"] = active_cycle
            worker_state.pop("active_outcome", None)
            worker_state["run_index"] = run_index
            worker_state["active_run_id"] = run_id
            worker_state["worker_stage"] = "ideation"
            worker_state["worker_stage_ts_utc"] = _utc_now_iso()
            _save_worker_runtime_state(worker_state_path, worker_state)

        _update_worker_info(
            worker_info_path,
            expert_spec=expert_spec,
            current_run_index=run_index,
            run_id=run_id,
        )

        parent_best_run_id = str(active_cycle.get("parent_best_run_id", "")).strip()
        baseline_commit = str(active_cycle.get("baseline_commit", "")).strip()
        baseline_metric_raw = active_cycle.get("baseline_metric")
        baseline_metric = float(baseline_metric_raw) if isinstance(baseline_metric_raw, (float, int)) else None
        seed_idea = str(active_cycle.get("seed_idea", "")).strip()

        idea_title = str(active_cycle.get("idea_title", "")).strip()
        idea_branch_name = str(active_cycle.get("idea_branch_name", "")).strip()
        idea_outline = str(active_cycle.get("idea_outline", "")).strip()
        aider_plan_prompt = str(active_cycle.get("aider_plan_prompt", "")).strip()
        aider_impl_prompt = str(active_cycle.get("aider_impl_prompt", "")).strip()
        feature_branch = str(active_cycle.get("feature_branch", "")).strip()

        worktree_path = _build_worker_worktree_path(output_root, worker_id, run_index)
        candidate_output_dir = _build_worker_output_dir(output_root, worker_id, run_index)
        run_info_path = candidate_output_dir / "run_info.json"
        candidate_output_dir.mkdir(parents=True, exist_ok=True)
        candidate_repo = worktree_path

        outcome = _deserialize_experiment_outcome(worker_state.get("active_outcome"))
        idea_payload: Any = None
        defer_feature_branch_delete = False
        cycle_completed = False

        _update_run_info(
            run_info_path,
            {
                "worker_id": worker_id,
                "worker_role": worker_role,
                "run_id": run_id,
                "run_index": int(run_index),
                "expert": _expert_payload_from_spec(expert_spec),
                "current_run_state": resumed_stage if resumed_stage else "ideation",
                "worker_stage": worker_state.get("worker_stage"),
                "active_cycle": dict(active_cycle),
            },
        )

        try:
            if resumed_stage not in {"executing", "evaluating"}:
                worker_state["worker_stage"] = "ideation"
                worker_state["worker_stage_ts_utc"] = _utc_now_iso()
                _touch_worker_heartbeat(worker_state_path, worker_state, phase="ideation")

                current_best_entry = _get_best_leaderboard_entry(
                    leaderboard_path=leaderboard_path,
                    lock_path=leaderboard_lock_path,
                    higher_is_better=config.metric_higher_is_better,
                )
                parent_best_run_id = ""
                if current_best_entry is not None:
                    parent_best_run_id = str(current_best_entry.get("run_id", "")).strip()

                baseline_commit, baseline_metric = _resolve_current_best_baseline_with_best_entry(
                    best_entry=current_best_entry,
                    config=config,
                    output_root=output_root,
                )

                if config.start_out_metric_baseline is not None:
                    if baseline_metric is None or baseline_metric <= config.start_out_metric_baseline:
                        baseline_metric = config.start_out_metric_baseline

                repo_root = Path(config.target_repo)
                files_for_prompt = _iter_candidate_files(
                    repo_root,
                    config.denylist_patterns,
                    config.aider_only_patterns,
                    include_aider_only=False,
                    allowed_suffixes=config.allowed_file_suffixes,
                )
                if not files_for_prompt:
                    raise RuntimeError("No editable files found for ideation prompt.")

                seed_idea = _read_text_if_exists(Path(config.input_idea_path)).strip()
                if not seed_idea:
                    seed_idea = "Improve training quality while keeping implementation simple and stable."
                user_priority_prompt = ""
                if config.user_prompt_file.strip():
                    user_priority_prompt = _read_text_if_exists(Path(config.user_prompt_file)).strip()

                experiments_md_path = output_root / config.experiments_md_name
                history_md = _read_text_if_exists(experiments_md_path)
                shared_memory_notes = _read_text_if_exists(output_root / "memory.md").strip()
                worker_memory_notes = _read_text_if_exists(worker_memory_path).strip()
                memory_notes = (
                    "## Shared Memory (all workers)\n"
                    f"{shared_memory_notes if shared_memory_notes else '(no shared memory.md notes available)'}\n\n"
                    f"## Worker Memory ({worker_id})\n"
                    f"{worker_memory_notes if worker_memory_notes else '(no worker memory.md notes available)'}"
                )
                codebase_glimpse = "\n".join([os.path.relpath(path, repo_root) for path in files_for_prompt][:100])
                file_summaries = _build_file_summaries_for_prompt(config, repo_root, files_for_prompt)
                stored_improvement_brief = worker_state.get("next_improvement_brief")
                improvement_brief = (
                    str(stored_improvement_brief).strip()
                    if isinstance(stored_improvement_brief, str) and str(stored_improvement_brief).strip()
                    else "(no prior run improvement brief yet; generate one after this run finishes)"
                )
                tried_ideas = _build_tried_ideas_summary(_load_leaderboard(leaderboard_path))

                expert_str = f" {expert_spec[0]} \n {expert_spec[1]} " if expert_spec is not None else ""

                websearch_ideas = ""
                if config.websearch_idea_prestep_enabled:
                    websearch_idea_prompt = _build_websearch_idea_prompt(
                        seed_idea=seed_idea,
                        user_priority_prompt=user_priority_prompt,
                        codebase_glimpse=codebase_glimpse,
                        experiments_history=history_md,
                        file_summaries=file_summaries,
                        improvement_brief=improvement_brief,
                        prior_stage_ideas=tried_ideas,
                        memory_notes=memory_notes,
                        ideas_count=10,
                        expert_str=expert_str,
                    )
                    _vprint(config, f"Websearch idea prompt:\n{websearch_idea_prompt}")

                    websearch_ideas_payload = _query_ideation_json(
                        config,
                        websearch_idea_prompt,
                        expert_str=expert_str,
                    )

                    _vprint(config, f"Websearch idea payload:\n{websearch_ideas_payload}")

                    websearch_ideas = _format_websearch_ideas_for_ideation_context(
                        websearch_ideas_payload,
                        max_items=10,
                    )

                    _vprint(config, f"Websearch ideas:\n{websearch_ideas}")

                else:
                    _vprint(config, "Websearch idea prestep disabled by config.")
                ideation_prompt = _build_ideation_prompt(
                    seed_idea=seed_idea,
                    user_priority_prompt=user_priority_prompt,
                    codebase_glimpse=codebase_glimpse,
                    experiments_history=history_md,
                    file_summaries=file_summaries,
                    improvement_brief=improvement_brief,
                    prior_stage_ideas=tried_ideas,
                    memory_notes=memory_notes,
                    ideas_count=1,
                    websearch_ideas=websearch_ideas,
                )
                if expert_spec is not None:
                    ideation_prompt = _extend_ideation_prompt_with_expert(
                        ideation_prompt,
                        expert_name=expert_spec[0],
                        expert_description=expert_spec[1],
                    )

                _vprint(config, f"Ideation prompt:\n{ideation_prompt}")

                idea_payload = _query_ideation_json(config, ideation_prompt, expert_str=expert_str)
                (
                    idea_title,
                    idea_branch_name,
                    idea_outline,
                    aider_plan_prompt,
                    aider_impl_prompt,
                ) = _normalize_idea_payload(_extract_idea_payload_list(idea_payload)[0])

                _vprint(config, f"Idea payload:\n{idea_payload}")

                feature_branch = _make_feature_branch_name(
                    idea_branch_name,
                    suffix=f"{worker_id}-r{run_index:06d}",
                )

                active_cycle.update(
                    {
                        "run_index": run_index,
                        "run_id": run_id,
                        "stage": "executing",
                        "parent_best_run_id": parent_best_run_id,
                        "baseline_commit": baseline_commit,
                        "baseline_metric": baseline_metric,
                        "seed_idea": seed_idea,
                        "idea_title": idea_title,
                        "idea_branch_name": idea_branch_name,
                        "idea_outline": idea_outline,
                        "aider_plan_prompt": aider_plan_prompt,
                        "aider_impl_prompt": aider_impl_prompt,
                        "feature_branch": feature_branch,
                    }
                )
                worker_state["active_cycle"] = active_cycle
                worker_state["worker_stage"] = "executing"
                worker_state["worker_stage_ts_utc"] = _utc_now_iso()
                _save_worker_runtime_state(worker_state_path, worker_state)
                resumed_stage = "executing"
                _update_run_info(
                    run_info_path,
                    {
                        "current_run_state": "executing",
                        "worker_stage": worker_state.get("worker_stage"),
                        "idea_payload": idea_payload,
                        "active_cycle": dict(active_cycle),
                    },
                )

            if resumed_stage == "evaluating" and outcome is None:
                # If evaluating was persisted but payload is missing, fall back to execute stage.
                resumed_stage = "executing"
                worker_state["worker_stage"] = "executing"
                worker_state["worker_stage_ts_utc"] = _utc_now_iso()
                _save_worker_runtime_state(worker_state_path, worker_state)
                _update_run_info(
                    run_info_path,
                    {
                        "current_run_state": "executing",
                        "worker_stage": worker_state.get("worker_stage"),
                        "active_cycle": dict(active_cycle),
                    },
                )

            if resumed_stage != "evaluating":
                worker_state["worker_stage"] = "executing"
                worker_state["worker_stage_ts_utc"] = _utc_now_iso()
                _touch_worker_heartbeat(worker_state_path, worker_state, phase="executing")

                if not worktree_path.exists():
                    candidate_repo = _create_candidate_worktree(
                        main_repo=config.target_repo,
                        worktree_path=worktree_path,
                        feature_branch=feature_branch,
                        baseline_commit=baseline_commit,
                    )
                else:
                    candidate_repo = worktree_path

                runtime_for_run = config.runtime
                runtime_source_path = candidate_repo / "config" / ".scian.yaml"
                runtime_source = "base config"
                try:
                    repo_config = _load_repo_config_yaml(str(runtime_source_path))
                    runtime_raw = repo_config.get("runtime")
                    if isinstance(runtime_raw, str) and runtime_raw.strip():
                        runtime_candidate = runtime_raw.strip()
                        _runtime_to_seconds(runtime_candidate)
                        runtime_for_run = runtime_candidate
                        runtime_source = str(runtime_source_path)
                except Exception as error:
                    logger.warning(
                        (
                            f"Worker {worker_id} failed to load runtime from {runtime_source_path}; "
                            f"falling back to configured runtime={config.runtime}: {error}"
                        )
                    )

                _vprint(
                    config,
                    f"Worker {worker_id} using runtime={runtime_for_run} (source={runtime_source})",
                )

                worker_config = replace(
                    config,
                    target_repo=str(candidate_repo),
                    runtime=runtime_for_run,
                    train_command=_resolve_train_command_for_repo(
                        config.train_command,
                        config.original_target_repo,
                        str(candidate_repo),
                    ),
                )
                files_to_edit = _iter_candidate_files(
                    candidate_repo,
                    worker_config.denylist_patterns,
                    worker_config.aider_only_patterns,
                    include_aider_only=True,
                    allowed_suffixes=worker_config.allowed_file_suffixes,
                )
                if not files_to_edit:
                    raise RuntimeError(f"No editable files found in worker repo: {candidate_repo}")

                outcome = _execute_plan(
                    config=worker_config,
                    files_to_edit=files_to_edit,
                    baseline_commit=baseline_commit,
                    feature_branch=feature_branch,
                    idea_title=idea_title,
                    idea_branch_name=idea_branch_name,
                    idea_outline=idea_outline,
                    aider_plan_prompt=aider_plan_prompt,
                    aider_impl_prompt=aider_impl_prompt,
                    baseline_metric_value=baseline_metric,
                    wandb_output_dir=candidate_output_dir / "wandb_pull",
                    candidate_output_dir=candidate_output_dir,
                    stage_index=run_index,
                    candidate_label=worker_id,
                    skip_aider=False,
                    heartbeat_callback=lambda: _touch_worker_heartbeat(
                        worker_state_path,
                        worker_state,
                        phase="cluster-poll",
                    ),
                )
                outcome.run_id = run_id
                outcome.worker_id = worker_id
                outcome.worker_role = worker_role

                worker_state["active_outcome"] = _serialize_experiment_outcome(outcome)
                worker_state["worker_stage"] = "evaluating"
                worker_state["worker_stage_ts_utc"] = _utc_now_iso()
                active_cycle["stage"] = "evaluating"
                worker_state["active_cycle"] = active_cycle
                _save_worker_runtime_state(worker_state_path, worker_state)
                run_state_snapshot = _load_trial_run_state(_build_trial_state_path(candidate_output_dir))
                _update_run_info(
                    run_info_path,
                    {
                        "current_run_state": "evaluating",
                        "worker_stage": worker_state.get("worker_stage"),
                        "active_cycle": dict(active_cycle),
                        "run_state": run_state_snapshot,
                        "experiment_outcome": _serialize_experiment_outcome(outcome),
                    },
                )

            if outcome is None:
                raise RuntimeError("Worker resumed in evaluating stage without a valid outcome payload.")

            worker_state["worker_stage"] = "evaluating"
            worker_state["worker_stage_ts_utc"] = _utc_now_iso()
            _touch_worker_heartbeat(worker_state_path, worker_state, phase="evaluating")

            # Summarize and persist outcome artifacts under lock-protected shared ledgers.
            if not candidate_repo.exists():
                candidate_repo = Path(config.target_repo)
            diff_patch = _extract_diff_patch(str(candidate_repo), outcome.baseline_commit, outcome.trial_commit)
            outcome.summary = _summarize_run_with_llm(
                config=config,
                outcome=outcome,
                diff_patch=diff_patch,
                decision_summary=outcome.summary,
            )
            if not _leaderboard_run_id_exists(leaderboard_path, run_id):
                _append_outcome_to_shared_logs(
                    config=config,
                    output_root=output_root,
                    lock_path=leaderboard_lock_path,
                    leaderboard_path=leaderboard_path,
                    outcome=outcome,
                    role=worker_role,
                    parent_entry_ids=([parent_best_run_id] if parent_best_run_id else []),
                )
            else:
                logger.info(f"Skipping duplicate leaderboard append for resumed run={run_id}")
            _touch_worker_heartbeat(worker_state_path, worker_state, phase="post-log")

            # Build the next-run improvement brief after this run finishes,
            # using this run's result payload and the assigned expert persona (if any).
            expert_name = expert_spec[0] if expert_spec is not None else None
            expert_description = expert_spec[1] if expert_spec is not None else None
            next_improvement_brief = _build_stage_improvement_ideas_with_websearch(
                config=config,
                stage_index=run_index,
                seed_idea=seed_idea,
                stage_baseline_metric=baseline_metric,
                merged_feature_branches=[],
                skipped_conflict_feature_branches=[],
                stage_outcomes=[outcome],
                expert_name=expert_name,
                expert_description=expert_description,
            )
            worker_state["next_improvement_brief"] = next_improvement_brief
            _save_worker_runtime_state(worker_state_path, worker_state)
            _update_run_info(
                run_info_path,
                {
                    "current_run_state": "completed",
                    "worker_stage": worker_state.get("worker_stage"),
                    "active_cycle": dict(active_cycle),
                    "next_improvement_brief": next_improvement_brief,
                },
            )

            with _file_lock(memory_lock_path):
                _update_stage_memory_with_llm(
                    config=config,
                    memory_path=output_root / "memory.md",
                    stage_index=run_index,
                    stage_summary_text=f"{worker_id} completed {run_id}",
                    stage_outcomes=[outcome],
                )

            _update_stage_memory_with_llm(
                config=config,
                memory_path=worker_memory_path,
                stage_index=run_index,
                stage_summary_text=f"{worker_id} completed successful run {run_id}",
                stage_outcomes=[outcome],
                expert_name=expert_name,
                expert_description=expert_description,
            )

            merge_processed = bool(active_cycle.get("merge_processed", False))
            if (
                not merge_processed
                and outcome.status == "finished"
                and outcome.unified_metric is not None
                and _metric_is_better(
                    outcome.unified_metric,
                    baseline_metric,
                    config.metric_higher_is_better,
                )
            ):
                # TODO push current branch before merge attempt, to ensure aider merge-resolution process can access it if needed
                push_ok, push_output = _push_to_origin(config.target_repo, feature_branch)

                merged_ok, merged_commit = _attempt_merge_on_base_branch(
                    config=config,
                    lock_path=repo_lock_path,
                    feature_branch=feature_branch,
                )
                if merged_ok:
                    outcome.kept = True
                    # Push merged branch to origin
                    push_ok, push_output = _push_to_origin(config.target_repo)
                    if push_ok:
                        logger.info(f"Pushed merged branch to origin: {push_output}")
                    else:
                        logger.warning(f"Failed to push to origin: {push_output}")
                    reused_outcome = _try_reuse_current_outcome_for_merged_validation(
                        config=config,
                        output_root=output_root,
                        leaderboard_lock_path=leaderboard_lock_path,
                        memory_lock_path=memory_lock_path,
                        leaderboard_path=leaderboard_path,
                        worker_id=worker_id,
                        merge_run_id=run_id,
                        merged_commit=merged_commit,
                        candidate_repo=candidate_repo,
                        outcome=outcome,
                    )
                    if not reused_outcome:
                        merged_validation_proc = _spawn_merged_validation_process(
                            config=config,
                            worker_id=worker_id,
                            merge_run_id=run_id,
                            merged_commit=merged_commit,
                            baseline_metric=baseline_metric,
                        )

                        _ensure_pending_merged_validation_entry(
                            worker_state,
                            {
                                "merge_run_id": run_id,
                                "merged_commit": merged_commit,
                                "baseline_metric": baseline_metric,
                                "pid": merged_validation_proc.pid,
                                "spawned_ts_utc": _utc_now_iso(),
                            },
                        )
                        _save_worker_runtime_state(worker_state_path, worker_state)
                        logger.info(
                            f"Spawned merged-validation process pid={merged_validation_proc.pid} for run={run_id}"
                        )
                else:
                    mergefix_proc = _spawn_aider_merge_resolution_process(
                        config=config,
                        worker_id=worker_id,
                        merge_run_id=run_id,
                        feature_branch=feature_branch,
                        baseline_metric=baseline_metric,
                    )
                    defer_feature_branch_delete = True
                    logger.info(
                        (
                            "Merge conflict detected; spawned aider merge-resolution process "
                            f"pid={mergefix_proc.pid} for run={run_id}"
                        )
                    )

                active_cycle["merge_processed"] = True
                worker_state["active_cycle"] = active_cycle
                _save_worker_runtime_state(worker_state_path, worker_state)

            cycle_completed = True
        except Exception as error:
            logger.exception(f"Worker {worker_id} cycle failed: {error}")
            worker_state["worker_stage_ts_utc"] = _utc_now_iso()
            _save_worker_runtime_state(worker_state_path, worker_state)
            _update_run_info(
                run_info_path,
                {
                    "current_run_state": "error",
                    "worker_stage": worker_state.get("worker_stage"),
                    "active_cycle": dict(active_cycle),
                    "error": str(error),
                },
            )
        finally:
            if cycle_completed:
                should_delete_feature_branch = (
                    (not config.keep_failed_feature_branches)
                    and (not defer_feature_branch_delete)
                    and bool(feature_branch)
                )
                if should_delete_feature_branch:
                    with _file_lock(repo_lock_path):
                        _git_try(config.target_repo, "branch", "-D", feature_branch)
                _cleanup_candidate_worktree(config.target_repo, worktree_path)
                worker_state.pop("active_cycle", None)
                worker_state.pop("active_outcome", None)
                worker_state["active_run_id"] = ""
                worker_state["worker_stage"] = "idle"
                worker_state["worker_stage_ts_utc"] = _utc_now_iso()
                worker_state["last_cycle_ts_utc"] = _utc_now_iso()
                _save_worker_runtime_state(worker_state_path, worker_state)
                _touch_worker_heartbeat(worker_state_path, worker_state, phase="cycle-end")
                _update_run_info(
                    run_info_path,
                    {
                        "worker_stage": worker_state.get("worker_stage"),
                        "cycle_completed": True,
                    },
                )
            else:
                worker_state["last_cycle_error_ts_utc"] = _utc_now_iso()
                _save_worker_runtime_state(worker_state_path, worker_state)
                _touch_worker_heartbeat(worker_state_path, worker_state, phase="cycle-error")

        if run_once:
            break


def _run_async_workers(config: LoopConfig) -> None:
    """Launch and supervise fully asynchronous worker processes."""
    output_root = _resolve_output_dir(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / config.state_file_name
    if not state_path.exists():
        _save_state(state_path, {})

    experts = _load_expert_specs(Path(config.experts_path))
    worker_count = max(1, int(config.async_worker_count))
    expert_count = min(worker_count, int(config.expert_worker_count), len(experts))

    worker_specs: list[tuple[int, str, tuple[str, str] | None]] = []
    for worker_index in range(worker_count):
        if worker_index < expert_count:
            worker_specs.append((worker_index, "expert", experts[worker_index]))
        else:
            worker_specs.append((worker_index, "general", None))

    if config.single_iteration:
        processes: list[multiprocessing.Process] = []
        for worker_index, role, expert_spec in worker_specs:
            process = multiprocessing.Process(
                target=_run_async_worker_loop,
                args=(config, worker_index, role, expert_spec, True),
                name=f"sciantist-{worker_index:02d}",
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        return

    active: dict[int, multiprocessing.Process] = {}
    for worker_index, role, expert_spec in worker_specs:
        process = multiprocessing.Process(
            target=_run_async_worker_loop,
            args=(config, worker_index, role, expert_spec, False),
            name=f"sciantist-{worker_index:02d}",
        )
        process.start()
        active[worker_index] = process

    while True:
        for worker_index, role, expert_spec in worker_specs:
            process = active.get(worker_index)
            if process is not None and process.is_alive():
                worker_id = f"worker_{worker_index:02d}"
                worker_state_path = _build_worker_state_path(output_root, worker_id)
                worker_state = _load_worker_runtime_state(worker_state_path)
                heartbeat_raw = worker_state.get("heartbeat_ts_utc")
                if isinstance(heartbeat_raw, str) and heartbeat_raw:
                    try:
                        heartbeat_dt = datetime.fromisoformat(heartbeat_raw)
                        if heartbeat_dt.tzinfo is None:
                            heartbeat_dt = heartbeat_dt.replace(tzinfo=timezone.utc)
                        age_seconds = (datetime.now(timezone.utc) - heartbeat_dt).total_seconds()
                        if age_seconds > float(config.worker_stale_timeout_seconds):
                            logger.warning(
                                (
                                    f"Worker {worker_index:02d} stale heartbeat ({int(age_seconds)}s). "
                                    "Terminating for takeover restart."
                                )
                            )
                            process.terminate()
                            process.join(timeout=10)
                            active.pop(worker_index, None)
                    except ValueError:
                        pass
                continue
            if process is not None and process.exitcode == 0:
                continue
            logger.warning(f"Worker {worker_index:02d} exited; restarting after backoff.")
            time.sleep(config.worker_restart_backoff_seconds)
            replacement = multiprocessing.Process(
                target=_run_async_worker_loop,
                args=(config, worker_index, role, expert_spec, False),
                name=f"sciantist-{worker_index:02d}",
            )
            replacement.start()
            active[worker_index] = replacement
        time.sleep(2)


def _create_isolated_base_worktree(
    original_repo: str,
    output_root: Path,
    branch_name: str,
) -> str:
    """Create an isolated full copy of the original repo to work in.

    This ensures the original repo is never modified directly.
    All experiments run in the created copy under output_root/base_worktree.

    Returns:
        Path to the created base worktree directory.
    """
    base_worktree_path = output_root / "base_worktree"

    if base_worktree_path.exists():
        logger.info(f"Removing existing base worktree at {base_worktree_path}")
        # raise RuntimeError(
        #     f"Base worktree path already exists: {base_worktree_path}. "
        #     "Please remove it manually if you want to reuse the existing worktree, "
        #     "or ensure no other process is using it and allow the loop to remove it automatically on next run."
        # )
        # shutil.rmtree(base_worktree_path, ignore_errors=True)
        return str(base_worktree_path)

    base_worktree_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying isolated base worktree from {original_repo} to {base_worktree_path}")

    # Full directory copy - ignores .gitignore but preserves .git contents as-is
    shutil.copytree(
        src=original_repo,
        dst=str(base_worktree_path),
        symlinks=True,
        ignore=None,
    )

    logger.info(f"Isolated base worktree copied: {base_worktree_path}")
    return str(base_worktree_path)


def _initialize_start_out_metric_baseline(config: LoopConfig, output_root: Path) -> LoopConfig:
    """Create baseline branch + synthetic finished entry when start baseline metric is configured."""
    baseline_metric = config.start_out_metric_baseline
    if baseline_metric is None:
        return config

    if not config.start_out_branch.strip():
        raise RuntimeError("start_out_branch must not be empty when start_out_metric_baseline is set.")

    baseline_branch = config.branch_name.strip() or config.start_out_branch.strip()
    if not baseline_branch.endswith("_baseline"):
        baseline_branch = f"{baseline_branch}_baseline"

    _git(config.target_repo, "checkout", config.start_out_branch)
    baseline_commit = _git(config.target_repo, "rev-parse", "HEAD")
    _git(config.target_repo, "branch", "-f", baseline_branch, baseline_commit)
    _git(config.target_repo, "checkout", baseline_branch)

    updated_config = replace(config, branch_name=baseline_branch)

    run_id = f"bootstrap-baseline-{_slugify(baseline_branch)}-{baseline_commit[:10]}"
    leaderboard_lock_path = _build_leaderboard_lock_path(output_root, updated_config)
    leaderboard_path = _build_leaderboard_path(output_root, updated_config.leaderboard_json_name)

    if not _leaderboard_run_id_exists(leaderboard_path, run_id):
        outcome = ExperimentOutcome(
            timestamp_utc=_utc_now_iso(),
            idea_title="bootstrap_start_out_metric_baseline",
            idea_branch_name="baseline_bootstrap",
            feature_branch=baseline_branch,
            idea_outline="Synthetic baseline seeded from start_out branch commit.",
            aider_plan_prompt="",
            aider_impl_prompt="",
            baseline_commit=baseline_commit,
            trial_commit=baseline_commit,
            job_id="baseline-bootstrap",
            wandb_project=updated_config.wandb_project,
            status="finished",
            runtime_seconds=0,
            unified_metric=float(baseline_metric),
            metric_histories={},
            baseline_metric=float(baseline_metric),
            metric_delta=0.0,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=True,
            summary=(
                "Synthetic baseline entry created before worker loop start using "
                "start_out_metric_baseline and current start_out commit."
            ),
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
            run_id=run_id,
            worker_id="bootstrap",
            worker_role="baseline",
        )
        _append_outcome_to_shared_logs(
            config=updated_config,
            output_root=output_root,
            lock_path=leaderboard_lock_path,
            leaderboard_path=leaderboard_path,
            outcome=outcome,
            role="baseline",
        )

    return updated_config


def run_loop(config: LoopConfig) -> None:
    """Run the autonomous scientist loop until interrupted or configured stop."""
    output_root = _resolve_output_dir(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _initialize_file_summary_cache_if_needed(config, output_root)
    _vprint(config, f"Using output directory: {output_root}")

    original_target_repo = (
        config.original_target_repo.strip() if config.original_target_repo.strip() else config.target_repo
    )

    # Create isolated base worktree so original repo is never modified directly
    isolated_repo = _create_isolated_base_worktree(
        original_repo=original_target_repo,
        output_root=output_root,
        branch_name=config.start_out_branch,
    )

    config = replace(
        config,
        target_repo=isolated_repo,
        original_target_repo=original_target_repo,
    )
    _vprint(config, f"Using isolated worktree as base repo: {isolated_repo}")

    config = _initialize_start_out_metric_baseline(config, output_root)

    _ensure_repo_ready(config)

    if config.async_worker_mode:
        logger.info(
            (
                "Starting fully async worker mode: "
                f"workers={config.async_worker_count}, expert_workers={config.expert_worker_count}"
            )
        )
        _run_async_workers(config)
        return


def main() -> int:
    """Parse arguments and start the autonomous scientist loop."""
    from sciantist.cli import main as _cli_main

    return _cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
