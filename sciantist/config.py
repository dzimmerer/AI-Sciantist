"""Configuration and result data models for Sciantist."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TARGET_REPO = "/home/zimmerer/ws/chexclip/"
DEFAULT_TRAIN_COMMAND = "/home/zimmerer/ws/chexclip/scripts/launch_slurm.sh"
DEFAULT_INPUT_IDEA_PATH = "/home/zimmerer/ws/plygrnd/inputs/scian.md"
DEFAULT_USER_PROMPT_FILE = ""
DEFAULT_EXPERTS_PATH = "./expert.md"
DEFAULT_BRANCH = "autoresearch/loop"
DEFAULT_START_OUT_BRANCH = DEFAULT_BRANCH
DEFAULT_RUNTIME = "05:00:00"
DEFAULT_POLL_SECONDS = 60
DEFAULT_FIX_ATTEMPTS = 10
DEFAULT_MAX_WANDB_RETRIES = 20
DEFAULT_TRACKING_BACKEND = "wandb"
DEFAULT_WANDB_PROJECT = "chexclip-sciantist"
DEFAULT_WANDB_ENTITY = ""
DEFAULT_OUTPUT_DIR = "./outputs/sciantist/"
DEFAULT_MAX_SUMMARY_FILES = 20
DEFAULT_MAX_FILE_SUMMARY_CHARS = 12000
DEFAULT_FILE_SUMMARY_CACHE_NAME = "file_summary_cache.json"
DEFAULT_INIT_CACHE_SUMMARY_FILE = ""
DEFAULT_IDEAS_PER_STAGE = 5
DEFAULT_WEBSEARCH_IDEA_PRESTEP_ENABLED = True
DEFAULT_ASYNC_WORKER_COUNT = 5
DEFAULT_EXPERT_WORKER_COUNT = 5
DEFAULT_WORKER_RESTART_BACKOFF_SECONDS = 5
DEFAULT_WORKER_HEARTBEAT_SECONDS = 60
DEFAULT_WORKER_STALE_TIMEOUT_SECONDS = 21600
DEFAULT_CLUSTER_RUNNER_SCRIPT = "/home/zimmerer/ws/clustermin/run_on_cluster.sh"
DEFAULT_STOP_BEFORE_SECS = 660
DEFAULT_START_OUT_METRIC_BASELINE: float | None = None
DEFAULT_METRIC_WEIGHTS: dict[str, float] = {
    "val/image_to_text_recall@10": 0.5,
    "val/text_to_image_recall@10": 0.5,
}

# DEFAULT_OAI_API_BASE = "http://0.0.0.0:8595/"
# DEFAULT_OAI_API_KEY = "sk-1234"
# DEFAULT_MODEL_NAME = "qwen3.5-122b"
# DEFAULT_AIDER_MODEL = f"openai/{DEFAULT_MODEL_NAME}"
DEFAULT_OAI_API_BASE = "https://api.minimax.io/"
DEFAULT_OAI_API_KEY = ""
DEFAULT_MODEL_NAME = "MiniMax-M2.7"
DEFAULT_AIDER_MODEL = f"openai/{DEFAULT_MODEL_NAME}"
DEFAULT_CODER_BACKEND = "aider"

STOP_BEFORE_SECS = 6000


@dataclass
class LoopConfig:
    """Configuration for the autonomous experiment loop."""

    target_repo: str = DEFAULT_TARGET_REPO
    original_target_repo: str = DEFAULT_TARGET_REPO
    train_command: str = DEFAULT_TRAIN_COMMAND
    input_idea_path: str = DEFAULT_INPUT_IDEA_PATH
    user_prompt_file: str = DEFAULT_USER_PROMPT_FILE
    experts_path: str = DEFAULT_EXPERTS_PATH
    start_out_branch: str = DEFAULT_START_OUT_BRANCH
    branch_name: str = DEFAULT_BRANCH
    runtime: str = DEFAULT_RUNTIME
    poll_seconds: int = DEFAULT_POLL_SECONDS
    max_fix_attempts: int = DEFAULT_FIX_ATTEMPTS
    max_wandb_retries: int = DEFAULT_MAX_WANDB_RETRIES
    tracking_backend: str = DEFAULT_TRACKING_BACKEND
    openai_api_base: str = DEFAULT_OAI_API_BASE
    openai_api_key_env: str = "MINIMAX_KEY"
    openai_api_key: str = DEFAULT_OAI_API_KEY
    model_name: str = DEFAULT_MODEL_NAME
    aider_model: str = DEFAULT_AIDER_MODEL
    coder_backend: str = DEFAULT_CODER_BACKEND
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_entity: str = DEFAULT_WANDB_ENTITY
    metric_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_METRIC_WEIGHTS))
    metric_higher_is_better: bool = True
    start_out_metric_baseline: float | None = DEFAULT_START_OUT_METRIC_BASELINE
    experiments_md_name: str = "experiments.md"
    experiments_tsv_name: str = "experiments.tsv"
    state_file_name: str = ".autoresearch_state.json"
    output_dir: str = DEFAULT_OUTPUT_DIR
    max_summary_files: int = DEFAULT_MAX_SUMMARY_FILES
    max_file_summary_chars: int = DEFAULT_MAX_FILE_SUMMARY_CHARS
    include_file_summaries: bool = True
    file_summary_cache_name: str = DEFAULT_FILE_SUMMARY_CACHE_NAME
    init_cache_summary_file: str = DEFAULT_INIT_CACHE_SUMMARY_FILE
    cluster_extra_args: str = ""
    cluster_target: str = "juwels"
    cluster_runner_script: str = DEFAULT_CLUSTER_RUNNER_SCRIPT
    allowed_file_suffixes: list[str] = field(default_factory=lambda: [".py", ".yaml", ".yml"])
    denylist_patterns: list[str] = field(
        default_factory=lambda: [
            r"(^|/)\.venv(/|$)",
            r"(^|/)n\.aider(/|$)",
            r"(^|/)aider(/|$)",
            r"(^|/)outputs(/|$)",
            r"(^|/)wandb(/|$)",
            r"(^|/)eval(/|$)",
            r"(^|/)evaluation(/|$)",
            r"(^|/)validation(/|$)",
        ]
    )
    aider_only_patterns: list[str] = field(default_factory=lambda: [r"(^|/)tests?(/|$)"])
    allow_dirty_repo: bool = False
    keep_failed_feature_branches: bool = True
    run_forever: bool = True
    dry_run: bool = False
    single_iteration: bool = False
    verbose: bool = True
    ideas_per_stage: int = DEFAULT_IDEAS_PER_STAGE
    websearch_idea_prestep_enabled: bool = DEFAULT_WEBSEARCH_IDEA_PRESTEP_ENABLED
    stage_multi_ideation_prompts: bool = False
    use_worktrees: bool = True
    candidate_output_subdirs: bool = True
    run_stage_baseline: bool = True
    async_worker_mode: bool = True
    async_worker_count: int = DEFAULT_ASYNC_WORKER_COUNT
    expert_worker_count: int = DEFAULT_EXPERT_WORKER_COUNT
    worker_restart_backoff_seconds: int = DEFAULT_WORKER_RESTART_BACKOFF_SECONDS
    worker_heartbeat_seconds: int = DEFAULT_WORKER_HEARTBEAT_SECONDS
    worker_stale_timeout_seconds: int = DEFAULT_WORKER_STALE_TIMEOUT_SECONDS
    stop_before_secs: int = DEFAULT_STOP_BEFORE_SECS
    lock_file_name: str = ".sciantist.lock"
    leaderboard_json_name: str = "leaderboard.json"


@dataclass
class ExperimentOutcome:
    """Result payload of one full experiment iteration."""

    timestamp_utc: str
    idea_title: str
    idea_branch_name: str
    feature_branch: str
    idea_outline: str
    aider_plan_prompt: str
    aider_impl_prompt: str
    baseline_commit: str
    trial_commit: str
    job_id: str
    wandb_project: str
    status: str
    runtime_seconds: int | None
    unified_metric: float | None
    metric_histories: dict[str, list[float]] | None
    baseline_metric: float | None
    metric_delta: float | None
    avg_gpu_util: float | None
    avg_gpu_memory: float | None
    kept: bool
    summary: str
    info_log_excerpt: str
    wandb_log_excerpt: str
    err_log_excerpt: str
    run_id: str = ""
    worker_id: str = ""
    worker_role: str = "general"
    merged_validation_of: str = ""
    parent_entry_ids: list[str] = field(default_factory=list)
    currently_in_best_path: bool = False


@dataclass
class StageOutcome:
    """Aggregate result payload for one stage execution."""

    stage_index: int
    baseline_commit: str
    baseline_metric: float | None
    merged_feature_branches: list[str]
    skipped_conflict_feature_branches: list[str]
    candidate_count: int
    new_base_commit: str
    best_metric_after_stage: float | None
    best_commit_after_stage: str | None
    stage_improvement_ideas: str
    summary: str


def _load_repo_config_yaml(config_path: str) -> dict[str, Any]:
    """Load optional per-repo YAML config. Missing file returns empty config."""
    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required for --repo-config support. Install with: uv sync") from error

    raw = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Repo config at {config_path} must be a YAML mapping.")
    return parsed


def _load_default_config_yaml(config_path: str = "config/default_config.yaml") -> dict[str, Any]:
    """Load optional default YAML config shared across repos."""
    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required for default_config.yaml support. Install with: uv sync") from error

    raw = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Default config at {config_path} must be a YAML mapping.")
    return parsed
