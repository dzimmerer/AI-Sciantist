"""CLI entrypoint for Sciantist."""

from __future__ import annotations

import argparse
import json
from typing import Any

from loguru import logger

from sciantist.config import (
    DEFAULT_AIDER_MODEL,
    DEFAULT_BRANCH,
    DEFAULT_CLUSTER_RUNNER_SCRIPT,
    DEFAULT_CODER_BACKEND,
    DEFAULT_EXPERTS_PATH,
    DEFAULT_FIX_ATTEMPTS,
    DEFAULT_IDEAS_PER_STAGE,
    DEFAULT_INPUT_IDEA_PATH,
    DEFAULT_MAX_FILE_SUMMARY_CHARS,
    DEFAULT_MAX_SUMMARY_FILES,
    DEFAULT_MAX_WANDB_RETRIES,
    DEFAULT_METRIC_WEIGHTS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OAI_API_BASE,
    DEFAULT_OAI_API_KEY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLL_SECONDS,
    DEFAULT_RUNTIME,
    DEFAULT_START_OUT_BRANCH,
    DEFAULT_TARGET_REPO,
    DEFAULT_TRACKING_BACKEND,
    DEFAULT_TRAIN_COMMAND,
    DEFAULT_USER_PROMPT_FILE,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_WEBSEARCH_IDEA_PRESTEP_ENABLED,
    LoopConfig,
    _load_default_config_yaml,
    _load_repo_config_yaml,
)


def _parse_metric_weights(weights_text: str) -> dict[str, float]:
    """Parse CLI metric weights from JSON text."""
    parsed = json.loads(weights_text)
    if not isinstance(parsed, dict):
        raise ValueError("metric weights must be a JSON object")
    result: dict[str, float] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            raise ValueError("metric weights keys must be strings and values numbers")
        result[key] = float(value)
    return result


def _coerce_metric_weights(weights_raw: Any) -> dict[str, float]:
    """Validate and normalize metric weights loaded from YAML."""
    if not isinstance(weights_raw, dict):
        raise ValueError("metric_weights in repo config must be a mapping")
    result: dict[str, float] = {}
    for key, value in weights_raw.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            raise ValueError("metric_weights keys must be strings and values numbers")
        result[key] = float(value)
    return result


def _coerce_optional_float(value: Any, field_name: str) -> float | None:
    """Convert optional scalar config values to float with clear errors."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be a number, got: {value!r}") from error
    raise ValueError(f"{field_name} must be a number or null, got: {type(value).__name__}")


def _resolve_setting(cli_value: Any, repo_config: dict[str, Any], repo_keys: list[str], default_value: Any) -> Any:
    """Resolve setting priority as CLI -> repo-config -> code default."""
    if cli_value is not None:
        return cli_value
    for key in repo_keys:
        if key in repo_config and repo_config[key] is not None:
            return repo_config[key]
    return default_value


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line parser for autonomous scientist options."""
    parser = argparse.ArgumentParser(description="Run autonomous scientist experiment loop.")
    parser.add_argument("--repo-config", default="config/.scian.yaml")
    parser.add_argument("--cluster-config", default="config/.scian-clusters.yaml")
    parser.add_argument("--cluster-name", default=None)
    parser.add_argument("--target-repo", default=None)
    parser.add_argument("--train-command", default=None)
    parser.add_argument("--input-idea-path", default=None)
    parser.add_argument("--user-prompt-file", default=None)
    parser.add_argument("--experts-path", default=None)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--poll-seconds", type=int, default=None)
    parser.add_argument("--stop-before-secs", type=int, default=None)
    parser.add_argument("--max-fix-attempts", type=int, default=None)
    parser.add_argument("--max-wandb-retries", type=int, default=None)
    parser.add_argument("--tracking-backend", default=None)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key-env", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--aider-model", default=None)
    parser.add_argument("--coder-backend", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--start-out-metric-baseline", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-summary-files", type=int, default=None)
    parser.add_argument("--max-file-summary-chars", type=int, default=None)
    parser.add_argument("--init-cache-summary-file", default=None)
    parser.add_argument("--no-file-summaries", action="store_true")
    parser.add_argument(
        "--metric-weights",
        default=None,
        help="JSON object of metric weights, e.g. {'m1':0.5,'m2':0.5}",
    )
    parser.add_argument("--metric-higher-is-better", dest="metric_higher_is_better", action="store_true", default=None)
    parser.add_argument("--metric-lower-is-better", dest="metric_higher_is_better", action="store_false")
    parser.add_argument("--deny-pattern", dest="deny_patterns", action="append", default=None)
    parser.add_argument("--aider-only-pattern", dest="aider_only_patterns", action="append", default=None)
    parser.add_argument("--allowed-file-suffix", dest="allowed_file_suffixes", action="append", default=None)
    parser.add_argument("--cluster-extra-args", default=None)
    parser.add_argument("--cluster-target", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--delete-failed-feature-branches", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--single-iteration", action="store_true")
    parser.add_argument("--no-forever", action="store_true")
    parser.add_argument("--ideas-per-stage", type=int, default=None)
    parser.add_argument(
        "--websearch-idea-prestep-enabled",
        dest="websearch_idea_prestep_enabled",
        action="store_true",
        default=None,
        help="Enable dedicated websearch-first idea generation before final ideation.",
    )
    parser.add_argument(
        "--no-websearch-idea-prestep-enabled",
        dest="websearch_idea_prestep_enabled",
        action="store_false",
        help="Disable dedicated websearch-first idea generation before final ideation.",
    )
    parser.add_argument(
        "--stage-multi-ideation-prompts",
        action="store_true",
        help=(
            "Use one ideation prompt per candidate in stage mode. "
            "Default behavior is one batched ideation prompt returning all candidates."
        ),
    )
    parser.add_argument("--no-worktrees", action="store_true")
    parser.add_argument("--no-candidate-output-subdirs", action="store_true")
    parser.add_argument("--no-stage-baseline", action="store_true")
    parser.add_argument("--async-worker-mode", dest="async_worker_mode", action="store_true", default=None)
    parser.add_argument("--no-async-worker-mode", dest="async_worker_mode", action="store_false")
    parser.add_argument("--async-worker-count", type=int, default=None)
    parser.add_argument("--expert-worker-count", type=int, default=None)
    parser.add_argument("--worker-restart-backoff-seconds", type=int, default=None)
    parser.add_argument("--worker-heartbeat-seconds", type=int, default=None)
    parser.add_argument("--worker-stale-timeout-seconds", type=int, default=None)
    parser.add_argument("--lock-file-name", default=None)
    parser.add_argument("--leaderboard-json-name", default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser


def _make_config_from_args(args: argparse.Namespace) -> LoopConfig:
    """Build validated loop config from parsed CLI arguments."""
    default_config = _load_default_config_yaml()
    repo_config = _load_repo_config_yaml(args.repo_config)
    effective_config = dict(default_config)
    effective_config.update(repo_config)
    defaults = LoopConfig()
    resolved_target_repo = str(
        _resolve_setting(args.target_repo, effective_config, ["target_repo"], DEFAULT_TARGET_REPO)
    )

    metric_weights = dict(DEFAULT_METRIC_WEIGHTS)
    if args.metric_weights is not None:
        metric_weights = _parse_metric_weights(args.metric_weights)
    elif "metric_weights" in effective_config and effective_config["metric_weights"] is not None:
        metric_weights = _coerce_metric_weights(effective_config["metric_weights"])

    deny_patterns: list[str]
    if args.deny_patterns is not None:
        deny_patterns = list(args.deny_patterns)
    else:
        yaml_deny = effective_config.get("deny_patterns", effective_config.get("deny_pattern"))
        if isinstance(yaml_deny, list) and all(isinstance(item, str) for item in yaml_deny):
            deny_patterns = list(yaml_deny)
        elif isinstance(yaml_deny, str):
            deny_patterns = [yaml_deny]
        else:
            deny_patterns = list(defaults.denylist_patterns)

    aider_only_patterns: list[str]
    if args.aider_only_patterns is not None:
        aider_only_patterns = list(args.aider_only_patterns)
    else:
        yaml_aider_only = effective_config.get(
            "aider_only_patterns",
            effective_config.get(
                "aider_only_pattern",
                effective_config.get("read_only_patterns", effective_config.get("read_only_pattern")),
            ),
        )
        if isinstance(yaml_aider_only, list) and all(isinstance(item, str) for item in yaml_aider_only):
            aider_only_patterns = list(yaml_aider_only)
        elif isinstance(yaml_aider_only, str):
            aider_only_patterns = [yaml_aider_only]
        else:
            aider_only_patterns = list(defaults.aider_only_patterns)

    allowed_file_suffixes: list[str]
    if args.allowed_file_suffixes is not None:
        allowed_file_suffixes = [str(item) for item in args.allowed_file_suffixes if str(item).strip()]
    else:
        yaml_suffixes = effective_config.get(
            "allowed_file_suffixes",
            effective_config.get("allowed_suffixes", effective_config.get("file_suffixes")),
        )
        if isinstance(yaml_suffixes, list) and all(isinstance(item, str) for item in yaml_suffixes):
            allowed_file_suffixes = [item for item in yaml_suffixes if item.strip()]
        elif isinstance(yaml_suffixes, str):
            allowed_file_suffixes = [yaml_suffixes]
        else:
            allowed_file_suffixes = list(defaults.allowed_file_suffixes)
    if not allowed_file_suffixes:
        allowed_file_suffixes = list(defaults.allowed_file_suffixes)

    resolved_branch_name = str(
        _resolve_setting(args.branch, effective_config, ["branch", "branch_name"], DEFAULT_BRANCH)
    )
    resolved_start_out_branch = str(
        _resolve_setting(
            None,
            effective_config,
            ["start_out_branch", "start_out", "start_branch"],
            resolved_branch_name or DEFAULT_START_OUT_BRANCH,
        )
    )
    resolved_start_out_metric_baseline = _coerce_optional_float(
        _resolve_setting(
            args.start_out_metric_baseline,
            effective_config,
            ["start_out_metric_baseline", "start_metric_baseline"],
            defaults.start_out_metric_baseline,
        ),
        "start_out_metric_baseline",
    )
    coder_backend_raw = _resolve_setting(
        args.coder_backend,
        effective_config,
        ["coder_backend"],
        DEFAULT_CODER_BACKEND,
    )
    coder_backend = str(coder_backend_raw).strip().lower()
    if coder_backend not in {"aider", "opencode"}:
        raise ValueError("coder_backend must be one of: aider, opencode")

    return LoopConfig(
        target_repo=resolved_target_repo,
        original_target_repo=resolved_target_repo,
        train_command=str(
            _resolve_setting(args.train_command, effective_config, ["train_command"], DEFAULT_TRAIN_COMMAND)
        ),
        input_idea_path=str(
            _resolve_setting(args.input_idea_path, effective_config, ["input_idea_path"], DEFAULT_INPUT_IDEA_PATH)
        ),
        user_prompt_file=str(
            _resolve_setting(
                args.user_prompt_file,
                effective_config,
                ["user_prompt_file", "user_prompt_path"],
                DEFAULT_USER_PROMPT_FILE,
            )
        ),
        experts_path=str(
            _resolve_setting(args.experts_path, effective_config, ["experts_path"], DEFAULT_EXPERTS_PATH)
        ),
        start_out_branch=resolved_start_out_branch,
        branch_name=resolved_branch_name,
        runtime=str(_resolve_setting(args.runtime, effective_config, ["runtime"], DEFAULT_RUNTIME)),
        poll_seconds=max(
            1, int(_resolve_setting(args.poll_seconds, effective_config, ["poll_seconds"], DEFAULT_POLL_SECONDS))
        ),
        stop_before_secs=max(
            0,
            int(
                _resolve_setting(
                    args.stop_before_secs,
                    effective_config,
                    ["stop_before_secs", "stop_before_seconds"],
                    defaults.stop_before_secs,
                )
            ),
        ),
        max_fix_attempts=max(
            0,
            int(
                _resolve_setting(
                    args.max_fix_attempts,
                    effective_config,
                    ["fix_attempts", "max_fix_attempts"],
                    DEFAULT_FIX_ATTEMPTS,
                )
            ),
        ),
        max_wandb_retries=max(
            0,
            int(
                _resolve_setting(
                    args.max_wandb_retries,
                    effective_config,
                    ["max_wandb_retries", "MAX_WANDB_RETRIES"],
                    DEFAULT_MAX_WANDB_RETRIES,
                )
            ),
        ),
        tracking_backend=str(
            _resolve_setting(
                args.tracking_backend,
                effective_config,
                ["tracking_backend"],
                DEFAULT_TRACKING_BACKEND,
            )
        ),
        openai_api_base=str(
            _resolve_setting(args.openai_api_base, effective_config, ["openai_api_base"], DEFAULT_OAI_API_BASE)
        ),
        openai_api_key_env=str(
            _resolve_setting(args.openai_api_key_env, effective_config, ["openai_api_key_env"], "MINIMAX_KEY")
        ),
        openai_api_key=str(
            _resolve_setting(args.openai_api_key, effective_config, ["openai_api_key"], DEFAULT_OAI_API_KEY)
        ),
        model_name=str(_resolve_setting(args.model_name, effective_config, ["model_name"], DEFAULT_MODEL_NAME)),
        aider_model=str(_resolve_setting(args.aider_model, effective_config, ["aider_model"], DEFAULT_AIDER_MODEL)),
        coder_backend=coder_backend,
        metric_weights=metric_weights,
        metric_higher_is_better=bool(
            _resolve_setting(
                args.metric_higher_is_better,
                effective_config,
                ["metric_higher_is_better"],
                defaults.metric_higher_is_better,
            )
        ),
        start_out_metric_baseline=resolved_start_out_metric_baseline,
        wandb_project=str(
            _resolve_setting(args.wandb_project, effective_config, ["wandb_project"], DEFAULT_WANDB_PROJECT)
        ),
        wandb_entity=str(
            _resolve_setting(args.wandb_entity, effective_config, ["wandb_entity"], defaults.wandb_entity)
        ),
        output_dir=str(_resolve_setting(args.output_dir, effective_config, ["output_dir"], DEFAULT_OUTPUT_DIR)),
        max_summary_files=max(
            1,
            int(
                _resolve_setting(
                    args.max_summary_files, effective_config, ["max_summary_files"], DEFAULT_MAX_SUMMARY_FILES
                )
            ),
        ),
        max_file_summary_chars=max(
            1000,
            int(
                _resolve_setting(
                    args.max_file_summary_chars,
                    effective_config,
                    ["max_file_summary_chars"],
                    DEFAULT_MAX_FILE_SUMMARY_CHARS,
                )
            ),
        ),
        include_file_summaries=(
            False
            if args.no_file_summaries
            else bool(_resolve_setting(None, effective_config, ["include_file_summaries"], True))
        ),
        init_cache_summary_file=str(
            _resolve_setting(
                args.init_cache_summary_file,
                effective_config,
                ["init_cache_summary_file", "initial_cache_summary_file"],
                defaults.init_cache_summary_file,
            )
        ),
        cluster_extra_args=str(
            _resolve_setting(
                args.cluster_extra_args,
                effective_config,
                ["cluster_extra_args"],
                defaults.cluster_extra_args,
            )
        ),
        cluster_target=str(
            _resolve_setting(args.cluster_target, effective_config, ["cluster_target"], defaults.cluster_target)
        ),
        cluster_runner_script=str(
            _resolve_setting(
                None,
                effective_config,
                ["cluster_runner_script"],
                DEFAULT_CLUSTER_RUNNER_SCRIPT,
            )
        ),
        allowed_file_suffixes=allowed_file_suffixes,
        denylist_patterns=deny_patterns,
        aider_only_patterns=aider_only_patterns,
        allow_dirty_repo=args.allow_dirty,
        keep_failed_feature_branches=not args.delete_failed_feature_branches,
        dry_run=args.dry_run,
        single_iteration=args.single_iteration,
        run_forever=not args.no_forever,
        verbose=args.verbose,
        ideas_per_stage=max(
            1,
            int(
                _resolve_setting(args.ideas_per_stage, effective_config, ["ideas_per_stage"], DEFAULT_IDEAS_PER_STAGE)
            ),
        ),
        websearch_idea_prestep_enabled=bool(
            _resolve_setting(
                args.websearch_idea_prestep_enabled,
                effective_config,
                ["websearch_idea_prestep_enabled", "websearch_idea_prestep"],
                DEFAULT_WEBSEARCH_IDEA_PRESTEP_ENABLED,
            )
        ),
        stage_multi_ideation_prompts=args.stage_multi_ideation_prompts,
        use_worktrees=not args.no_worktrees,
        candidate_output_subdirs=not args.no_candidate_output_subdirs,
        run_stage_baseline=not args.no_stage_baseline,
        async_worker_mode=bool(
            _resolve_setting(
                args.async_worker_mode,
                effective_config,
                ["async_worker_mode"],
                defaults.async_worker_mode,
            )
        ),
        async_worker_count=max(
            1,
            int(
                _resolve_setting(
                    args.async_worker_count,
                    effective_config,
                    ["async_worker_count"],
                    defaults.async_worker_count,
                )
            ),
        ),
        expert_worker_count=max(
            0,
            int(
                _resolve_setting(
                    args.expert_worker_count,
                    effective_config,
                    ["expert_worker_count"],
                    defaults.expert_worker_count,
                )
            ),
        ),
        worker_restart_backoff_seconds=max(
            1,
            int(
                _resolve_setting(
                    args.worker_restart_backoff_seconds,
                    effective_config,
                    ["worker_restart_backoff_seconds"],
                    defaults.worker_restart_backoff_seconds,
                )
            ),
        ),
        worker_heartbeat_seconds=max(
            5,
            int(
                _resolve_setting(
                    args.worker_heartbeat_seconds,
                    effective_config,
                    ["worker_heartbeat_seconds"],
                    defaults.worker_heartbeat_seconds,
                )
            ),
        ),
        worker_stale_timeout_seconds=max(
            60,
            int(
                _resolve_setting(
                    args.worker_stale_timeout_seconds,
                    effective_config,
                    ["worker_stale_timeout_seconds"],
                    defaults.worker_stale_timeout_seconds,
                )
            ),
        ),
        lock_file_name=str(
            _resolve_setting(
                args.lock_file_name,
                effective_config,
                ["lock_file_name"],
                defaults.lock_file_name,
            )
        ),
        leaderboard_json_name=str(
            _resolve_setting(
                args.leaderboard_json_name,
                effective_config,
                ["leaderboard_json_name"],
                defaults.leaderboard_json_name,
            )
        ),
    )


def main() -> int:
    """Parse arguments and start the autonomous scientist loop."""
    from scian import (
        _resolve_output_dir,
        _vprint,
        configure_active_cluster,
        configure_logging,
        run_loop,
    )

    parser = _build_arg_parser()
    args = parser.parse_args()
    config = _make_config_from_args(args)
    output_root = _resolve_output_dir(config.output_dir)
    log_path = configure_logging(output_root, verbose=config.verbose)
    logger.info(f"Loguru configured with file sink at {log_path}")
    selected_cluster_name = args.cluster_name or config.cluster_target
    profile = configure_active_cluster(args.cluster_config, selected_cluster_name)
    if config.verbose:
        _vprint(
            config,
            (
                f"Using cluster profile name={profile.name} target={profile.cluster_target} "
                f"scheduler={profile.scheduler} ssh={profile.ssh_target}"
            ),
        )
    run_loop(config)
    return 0
