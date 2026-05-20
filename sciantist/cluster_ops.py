"""Cluster and remote log helpers used by the autonomous loop."""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SSH_TARGET = "zimmerer1@juwels-booster.fz-juelich.de"
CLUSTER_RUNNER_SCRIPT = "/home/zimmerer/ws/clustermin/run_on_cluster.sh"
CLUSTER_TARGET = "juwels"
LOG_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9._+-]+$")
RUNTIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")

_SQUEUE_STATE_MAP = {
    "PENDING": "pending",
    "RUNNING": "running",
    "COMPLETING": "running",
    "CONFIGURING": "running",
    "RESIZING": "running",
    "SUSPENDED": "pending",
}

_SACCT_STATE_MAP = {
    "PENDING": "pending",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "finished",
    "FAILED": "crashed",
    "CANCELLED": "crashed",
    "TIMEOUT": "crashed",
    "PREEMPTED": "crashed",
    "BOOT_FAIL": "crashed",
    "DEADLINE": "crashed",
    "OUT_OF_MEMORY": "crashed",
    "NODE_FAIL": "crashed",
    "REVOKED": "crashed",
}

_BJOBS_STATE_MAP = {
    "PEND": "pending",
    "PROV": "pending",
    "PSUSP": "pending",
    "USUSP": "pending",
    "SSUSP": "pending",
    "WAIT": "pending",
    "RUN": "running",
    "DONE": "finished",
    "EXIT": "crashed",
    "ZOMBI": "crashed",
}


@dataclass(frozen=True)
class ClusterProfile:
    """Resolved runtime settings for one cluster entry."""

    name: str
    cluster_target: str
    ssh_target: str
    scheduler: str
    info_log_path: str
    error_log_path: str
    submit_extra_args: str
    wandb_sync: bool


_DEFAULT_CLUSTER_PROFILE = ClusterProfile(
    name="juwels",
    cluster_target=CLUSTER_TARGET,
    ssh_target=SSH_TARGET,
    scheduler="slurm",
    info_log_path="$HOME/logs/{log_id}.out",
    error_log_path="$HOME/logs/{log_id}.err",
    submit_extra_args="",
    wandb_sync=True,
)

_active_cluster_profile = _DEFAULT_CLUSTER_PROFILE


def get_active_cluster_profile() -> ClusterProfile:
    """Return the currently configured cluster profile."""
    return _active_cluster_profile


def _runtime_to_seconds(runtime: str) -> int:
    """Convert HH:MM:SS runtime string to seconds."""
    parts = runtime.strip().split(":")
    if len(parts) != 3:
        raise ValueError("runtime must be HH:MM:SS")
    hours, minutes, seconds = (int(part) for part in parts)
    return hours * 3600 + minutes * 60 + seconds


def _coerce_cluster_profile(name: str, raw: dict[str, Any]) -> ClusterProfile:
    """Validate and build a cluster profile from YAML mapping."""
    if not isinstance(raw, dict):
        raise ValueError(f"Cluster profile '{name}' must be a mapping")

    scheduler_raw = str(raw.get("scheduler", "slurm")).strip().lower()
    if scheduler_raw not in {"slurm", "lsf"}:
        raise ValueError(f"Cluster profile '{name}' has invalid scheduler '{scheduler_raw}' (expected slurm or lsf)")

    cluster_target = str(raw.get("cluster_target", name)).strip()
    ssh_target = str(raw.get("ssh_target", SSH_TARGET)).strip()
    info_log_path = str(raw.get("info_log_path", "$HOME/logs/{log_id}.out"))
    error_log_path = str(raw.get("error_log_path", "$HOME/logs/{log_id}.err"))
    submit_extra_args = str(raw.get("submit_extra_args", "")).strip()
    wandb_sync_raw = raw.get("wandb_sync", True)
    if isinstance(wandb_sync_raw, bool):
        wandb_sync = wandb_sync_raw
    elif isinstance(wandb_sync_raw, str):
        normalized = wandb_sync_raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            wandb_sync = True
        elif normalized in {"0", "false", "no", "off"}:
            wandb_sync = False
        else:
            raise ValueError(
                f"Cluster profile '{name}' has invalid wandb_sync '{wandb_sync_raw}' (expected true/false)."
            )
    else:
        raise ValueError(
            f"Cluster profile '{name}' has invalid wandb_sync type '{type(wandb_sync_raw).__name__}' (expected bool)."
        )
    return ClusterProfile(
        name=name,
        cluster_target=cluster_target,
        ssh_target=ssh_target,
        scheduler=scheduler_raw,
        info_log_path=info_log_path,
        error_log_path=error_log_path,
        submit_extra_args=submit_extra_args,
        wandb_sync=wandb_sync,
    )


def load_cluster_catalog_yaml(config_path: str) -> dict[str, ClusterProfile]:
    """Load cluster catalog YAML file. Missing file returns default catalog."""
    path = Path(config_path)
    if not path.exists():
        return {_DEFAULT_CLUSTER_PROFILE.name: _DEFAULT_CLUSTER_PROFILE}

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required for --cluster-config support. Install with: uv sync") from error

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if parsed is None:
        return {_DEFAULT_CLUSTER_PROFILE.name: _DEFAULT_CLUSTER_PROFILE}
    if not isinstance(parsed, dict):
        raise ValueError(f"Cluster config at {config_path} must be a YAML mapping")

    clusters_raw = parsed.get("clusters")
    if not isinstance(clusters_raw, dict):
        raise ValueError(f"Cluster config at {config_path} must define a 'clusters' mapping")

    catalog: dict[str, ClusterProfile] = {}
    for name, raw in clusters_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Cluster names must be non-empty strings")
        catalog[name] = _coerce_cluster_profile(name, raw)

    if not catalog:
        raise ValueError(f"Cluster config at {config_path} has no cluster entries")
    return catalog


def configure_active_cluster(config_path: str, cluster_name: str | None = None) -> ClusterProfile:
    """Select active cluster profile from catalog by name."""
    global _active_cluster_profile
    catalog = load_cluster_catalog_yaml(config_path)
    selected_name = cluster_name or next(iter(catalog.keys()))
    if selected_name not in catalog:
        available = ", ".join(sorted(catalog.keys()))
        raise ValueError(f"Unknown cluster '{selected_name}'. Available clusters: {available}")
    _active_cluster_profile = catalog[selected_name]
    return _active_cluster_profile


def _run_ssh_command(ssh_target: str, remote_command: str) -> subprocess.CompletedProcess[str]:
    """Run a command on a remote machine via SSH and return the process result."""
    try:
        return subprocess.run(
            ["ssh", ssh_target, remote_command],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ssh command not found on this machine.") from exc


def _wrap_remote_login_shell(command: str) -> str:
    """Execute a remote command via a login shell to load site/user environment."""
    return f"bash -lc {shlex.quote(command)}"


def load_remote_log(log_id: str, ssh_target: str | None = None, log: str = "ERROR") -> str:
    """Load remote INFO/ERROR log content via SSH using active cluster profile."""
    if not LOG_ID_PATTERN.fullmatch(log_id):
        raise ValueError("Invalid log id. Use only letters, numbers, dot, underscore, and hyphen.")

    if log not in {"INFO", "ERROR"}:
        raise ValueError("Invalid log type. Use 'INFO' or 'ERROR'.")

    profile = get_active_cluster_profile()
    resolved_ssh_target = ssh_target or profile.ssh_target
    log_template = profile.info_log_path if log == "INFO" else profile.error_log_path
    remote_path = log_template.replace("{log_id}", log_id)
    remote_command = f"cat {remote_path}"
    try:
        result = _run_ssh_command(resolved_ssh_target, remote_command)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "no stderr output"
        raise RuntimeError(f"Failed to load remote log '{log_id}': {stderr}") from exc

    return result.stdout


def submit_cluster_job(
    command: str,
    runtime: str,
    extra_args: str = "",
    cluster_target: str | None = None,
    runner_script: str = CLUSTER_RUNNER_SCRIPT,
) -> int:
    """Submit a cluster job and return the job id parsed from the last output line."""
    command_value = command.strip()
    if not command_value:
        raise ValueError("command must not be empty.")

    runtime_value = runtime.strip()
    if not RUNTIME_PATTERN.fullmatch(runtime_value):
        raise ValueError("runtime must use HH:MM:SS format.")

    resolved_cluster_target = cluster_target or get_active_cluster_profile().cluster_target
    profile = get_active_cluster_profile()

    if profile.scheduler == "slurm":
        runtime_arg = f"--time {runtime_value}"
    elif profile.scheduler == "lsf":
        hours, minutes, _seconds = runtime_value.split(":")
        runtime_arg = f"-W {hours}:{minutes}"
    else:
        raise ValueError(f"Unsupported scheduler '{profile.scheduler}'.")

    x_parts = [runtime_arg]
    cluster_submit_args = profile.submit_extra_args.strip()
    extra_args_value = extra_args.strip()
    if extra_args_value:
        x_parts.append(extra_args_value)
    x_value = " ".join(x_parts)

    runner_args = [
        "sh",
        runner_script,
        "-t",
        resolved_cluster_target,
        "-c",
        command_value,
        "-x",
        x_value,
    ]
    if cluster_submit_args:
        runner_args.extend(shlex.split(cluster_submit_args))

    try:
        result = subprocess.run(
            runner_args,
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("sh command not found on this machine.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "no stderr output"
        raise RuntimeError(f"Failed to submit cluster job: {stderr}") from exc

    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("Failed to submit cluster job: no output returned.")

    # Parse job id from the last line only.
    job_id_text = output_lines[-1]
    if job_id_text.isdigit():
        return int(job_id_text)

    job_id_regex = re.compile(r"(?:Started\s+job\s+with\s+ID:\s*|Job\s*<)(\d+)(?:>)?")
    match = job_id_regex.search(job_id_text)
    if match:
        return int(match.group(1))

    raise RuntimeError(
        f"Failed to submit cluster job: unable to parse job id from last output line. Last line was '{job_id_text}'."
    )


def get_slurm_job_status(job_id: str | int, ssh_target: str = SSH_TARGET) -> str:
    """Return a normalized SLURM job status for a job id via SSH."""
    job_id_str = str(job_id).strip()
    if not job_id_str:
        raise ValueError("job_id must not be empty.")
    if not JOB_ID_PATTERN.fullmatch(job_id_str):
        raise ValueError("Invalid job id. Use only letters, numbers, dot, underscore, plus, and hyphen.")

    try:
        squeue_result = _run_ssh_command(
            ssh_target,
            f"squeue --noheader --format=%T --jobs {job_id_str}",
        )
    except subprocess.CalledProcessError:
        squeue_result = None

    if squeue_result and squeue_result.stdout.strip():
        state = squeue_result.stdout.strip().splitlines()[0].split()[0].upper()
        if state in _SQUEUE_STATE_MAP:
            return _SQUEUE_STATE_MAP[state]

    try:
        sacct_result = _run_ssh_command(
            ssh_target,
            f"sacct -n -X -j {job_id_str} --format=State",
        )
    except subprocess.CalledProcessError:
        return "unknown"

    states = [line.strip().split()[0].rstrip("+").upper() for line in sacct_result.stdout.splitlines() if line.strip()]
    for state in states:
        if state in _SACCT_STATE_MAP:
            return _SACCT_STATE_MAP[state]

    return "unknown"


def get_lsf_job_status(job_id: str | int, ssh_target: str = SSH_TARGET) -> str:
    """Return a normalized LSF job status for a job id via SSH."""
    job_id_str = str(job_id).strip()
    if not job_id_str:
        raise ValueError("job_id must not be empty.")
    if not JOB_ID_PATTERN.fullmatch(job_id_str):
        raise ValueError("Invalid job id. Use only letters, numbers, dot, underscore, plus, and hyphen.")

    try:
        bjobs_result = _run_ssh_command(
            ssh_target,
            _wrap_remote_login_shell(f"bjobs -noheader -o stat {job_id_str}"),
        )
    except subprocess.CalledProcessError:
        bjobs_result = None

    if bjobs_result and bjobs_result.stdout.strip():
        state = bjobs_result.stdout.strip().splitlines()[0].split()[0].upper()
        if state in _BJOBS_STATE_MAP:
            return _BJOBS_STATE_MAP[state]

    try:
        bhist_result = _run_ssh_command(
            ssh_target,
            _wrap_remote_login_shell(f"bhist -n 1 -noheader -o stat {job_id_str}"),
        )
    except subprocess.CalledProcessError:
        return "unknown"

    states = [line.strip().split()[0].upper() for line in bhist_result.stdout.splitlines() if line.strip()]
    for state in states:
        if state in _BJOBS_STATE_MAP:
            return _BJOBS_STATE_MAP[state]

    return "unknown"


def _parse_slurm_duration_to_seconds(duration: str) -> int | None:
    """Convert a SLURM duration string to seconds."""
    value = duration.strip()
    if not value or value in {"N/A", "Unknown"}:
        return None

    days = 0
    time_part = value
    if "-" in value:
        day_part, time_part = value.split("-", maxsplit=1)
        if not day_part.isdigit():
            return None
        days = int(day_part)

    parts = time_part.split(":")
    if len(parts) == 3:
        hours_str, minutes_str, seconds_str = parts
    elif len(parts) == 2:
        hours_str = "0"
        minutes_str, seconds_str = parts
    else:
        return None

    if not (hours_str.isdigit() and minutes_str.isdigit() and seconds_str.isdigit()):
        return None

    hours = int(hours_str)
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def get_slurm_job_runtime(job_id: str | int, ssh_target: str = SSH_TARGET) -> int | None:
    """Return SLURM job runtime in seconds via SSH, if available."""
    job_id_str = str(job_id).strip()
    if not job_id_str:
        raise ValueError("job_id must not be empty.")
    if not JOB_ID_PATTERN.fullmatch(job_id_str):
        raise ValueError("Invalid job id. Use only letters, numbers, dot, underscore, plus, and hyphen.")

    try:
        squeue_result = _run_ssh_command(
            ssh_target,
            f"squeue --noheader --format=%M --jobs {job_id_str}",
        )
    except subprocess.CalledProcessError:
        squeue_result = None

    if squeue_result and squeue_result.stdout.strip():
        elapsed = squeue_result.stdout.strip().splitlines()[0].strip()
        parsed = _parse_slurm_duration_to_seconds(elapsed)
        if parsed is not None:
            return parsed

    try:
        sacct_result = _run_ssh_command(
            ssh_target,
            f"sacct -n -X -j {job_id_str} --format=ElapsedRaw",
        )
    except subprocess.CalledProcessError:
        return None

    for line in sacct_result.stdout.splitlines():
        value = line.strip()
        if value.isdigit():
            return int(value)

    return None


def _parse_lsf_duration_to_seconds(duration: str) -> int | None:
    """Convert an LSF duration string to seconds."""
    value = duration.strip()
    if not value or value in {"N/A", "Unknown", "-"}:
        return None

    if value.isdigit():
        return int(value)

    parsed_hms = _parse_slurm_duration_to_seconds(value)
    if parsed_hms is not None:
        return parsed_hms

    lower_value = value.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", lower_value)
    if not matches:
        return None

    total_seconds = 0.0
    for amount_text, unit_text in matches:
        amount = float(amount_text)
        if unit_text.startswith("sec"):
            total_seconds += amount
        elif unit_text.startswith("min"):
            total_seconds += amount * 60
        elif unit_text.startswith("hour") or unit_text.startswith("hr"):
            total_seconds += amount * 3600
        elif unit_text.startswith("day"):
            total_seconds += amount * 86400
        else:
            return None

    return int(total_seconds)


def get_lsf_job_runtime(job_id: str | int, ssh_target: str = SSH_TARGET) -> int | None:
    """Return LSF job runtime in seconds via SSH, if available."""
    job_id_str = str(job_id).strip()
    if not job_id_str:
        raise ValueError("job_id must not be empty.")
    if not JOB_ID_PATTERN.fullmatch(job_id_str):
        raise ValueError("Invalid job id. Use only letters, numbers, dot, underscore, plus, and hyphen.")

    try:
        bjobs_result = _run_ssh_command(
            ssh_target,
            _wrap_remote_login_shell(f"bjobs -noheader -o run_time {job_id_str}"),
        )
    except subprocess.CalledProcessError:
        bjobs_result = None

    if bjobs_result and bjobs_result.stdout.strip():
        runtime_text = bjobs_result.stdout.strip().splitlines()[0].strip()
        parsed = _parse_lsf_duration_to_seconds(runtime_text)
        if parsed is not None:
            return parsed

    try:
        bhist_result = _run_ssh_command(
            ssh_target,
            _wrap_remote_login_shell(f"bhist -n 1 -noheader -o run_time {job_id_str}"),
        )
    except subprocess.CalledProcessError:
        return None

    for line in bhist_result.stdout.splitlines():
        parsed = _parse_lsf_duration_to_seconds(line)
        if parsed is not None:
            return parsed

    return None


def get_cluster_job_status(job_id: str | int, scheduler: str | None = None, ssh_target: str | None = None) -> str:
    """Return job status using active cluster scheduler unless explicitly overridden."""
    profile = get_active_cluster_profile()
    resolved_scheduler = (scheduler or profile.scheduler).strip().lower()
    resolved_ssh = ssh_target or profile.ssh_target
    if resolved_scheduler == "slurm":
        return get_slurm_job_status(job_id, ssh_target=resolved_ssh)
    if resolved_scheduler == "lsf":
        return get_lsf_job_status(job_id, ssh_target=resolved_ssh)
    raise ValueError(f"Unsupported scheduler '{resolved_scheduler}'.")


def get_cluster_job_runtime(
    job_id: str | int, scheduler: str | None = None, ssh_target: str | None = None
) -> int | None:
    """Return job runtime using active cluster scheduler unless explicitly overridden."""
    profile = get_active_cluster_profile()
    resolved_scheduler = (scheduler or profile.scheduler).strip().lower()
    resolved_ssh = ssh_target or profile.ssh_target
    if resolved_scheduler == "slurm":
        return get_slurm_job_runtime(job_id, ssh_target=resolved_ssh)
    if resolved_scheduler == "lsf":
        return get_lsf_job_runtime(job_id, ssh_target=resolved_ssh)
    raise ValueError(f"Unsupported scheduler '{resolved_scheduler}'.")
