"""Utility helpers for SCIEN workflows."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml

SSH_TARGET = "zimmerer1@juwels-booster.fz-juelich.de"
_DEFAULT_CLUSTER_RUNNER_SCRIPT = "/home/zimmerer/ws/clustermin/run_on_cluster.sh"
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


def _load_default_cluster_runner_script(config_path: str = "config/default_config.yaml") -> str:
    """Load cluster runner script path from default YAML config with fallback."""
    path = Path(config_path)
    if not path.exists():
        return _DEFAULT_CLUSTER_RUNNER_SCRIPT

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        return _DEFAULT_CLUSTER_RUNNER_SCRIPT

    value = parsed.get("cluster_runner_script")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _DEFAULT_CLUSTER_RUNNER_SCRIPT


CLUSTER_RUNNER_SCRIPT = _load_default_cluster_runner_script()


def load_remote_log(log_id: str, ssh_target: str = SSH_TARGET, log: str = "ERROR") -> str:
    """Load `$HOME/logs/{log_id}.err` from a remote host via SSH."""
    if not LOG_ID_PATTERN.fullmatch(log_id):
        raise ValueError("Invalid log id. Use only letters, numbers, dot, underscore, and hyphen.")

    if log not in {"INFO", "ERROR"}:
        raise ValueError("Invalid log type. Use 'INFO' or 'ERROR'.")

    if log == "INFO":
        remote_command = f"cat $HOME/logs/{log_id}.out"
    else:
        remote_command = f"cat $HOME/logs/{log_id}.err"
    try:
        result = _run_ssh_command(ssh_target, remote_command)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "no stderr output"
        raise RuntimeError(f"Failed to load remote log '{log_id}': {stderr}") from exc

    return result.stdout


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


def submit_cluster_job(
    command: str,
    runtime: str,
    extra_args: str = "",
    cluster_target: str = CLUSTER_TARGET,
    runner_script: str = CLUSTER_RUNNER_SCRIPT,
) -> int:
    """Submit a cluster job and return the job id parsed from the last output line."""
    command_value = command.strip()
    if not command_value:
        raise ValueError("command must not be empty.")

    runtime_value = runtime.strip()
    if not RUNTIME_PATTERN.fullmatch(runtime_value):
        raise ValueError("runtime must use HH:MM:SS format.")

    x_value = f"--time {runtime_value}"
    extra_args_value = extra_args.strip()
    if extra_args_value:
        x_value = f"{x_value} {extra_args_value}"

    try:
        result = subprocess.run(
            [
                "sh",
                runner_script,
                "-t",
                cluster_target,
                "-c",
                command_value,
                "-x",
                x_value,
            ],
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

    job_id_text = output_lines[-1]
    if not job_id_text.isdigit():
        raise RuntimeError(
            f"Failed to submit cluster job: last output line is not an integer job id (got '{job_id_text}')."
        )

    return int(job_id_text)


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


def print_slurm_job_status(job_id: str | int, ssh_target: str = SSH_TARGET) -> None:
    """Print the normalized SLURM status for a job id via SSH."""
    status = get_slurm_job_status(job_id, ssh_target=ssh_target)
    print(status)


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
            f"bjobs -noheader -o stat {job_id_str}",
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
            f"bhist -n 1 -noheader -o stat {job_id_str}",
        )
    except subprocess.CalledProcessError:
        return "unknown"

    states = [line.strip().split()[0].upper() for line in bhist_result.stdout.splitlines() if line.strip()]
    for state in states:
        if state in _BJOBS_STATE_MAP:
            return _BJOBS_STATE_MAP[state]

    return "unknown"


def print_lsf_job_status(job_id: str | int, ssh_target: str = SSH_TARGET) -> None:
    """Print the normalized LSF status for a job id via SSH."""
    status = get_lsf_job_status(job_id, ssh_target=ssh_target)
    print(status)


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


def _format_seconds_hms(total_seconds: int) -> str:
    """Format seconds into HH:MM:SS for display."""
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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


def print_slurm_job_runtime(job_id: str | int, ssh_target: str = SSH_TARGET) -> None:
    """Print SLURM job runtime in HH:MM:SS, or 'unknown' if unavailable."""
    runtime_seconds = get_slurm_job_runtime(job_id, ssh_target=ssh_target)
    if runtime_seconds is None:
        print("unknown")
        return
    print(_format_seconds_hms(runtime_seconds))


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
            f"bjobs -noheader -o run_time {job_id_str}",
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
            f"bhist -n 1 -noheader -o run_time {job_id_str}",
        )
    except subprocess.CalledProcessError:
        return None

    for line in bhist_result.stdout.splitlines():
        parsed = _parse_lsf_duration_to_seconds(line)
        if parsed is not None:
            return parsed

    return None


def print_lsf_job_runtime(job_id: str | int, ssh_target: str = SSH_TARGET) -> None:
    """Print LSF job runtime in HH:MM:SS, or 'unknown' if unavailable."""
    runtime_seconds = get_lsf_job_runtime(job_id, ssh_target=ssh_target)
    if runtime_seconds is None:
        print("unknown")
        return
    print(_format_seconds_hms(runtime_seconds))


def main() -> int:
    """Parse CLI arguments and either load a log or query job scheduler data."""
    parser = argparse.ArgumentParser(
        description=(
            "For a given ID, either load remote log file "
            f"'$HOME/logs/<id>.err' from {SSH_TARGET} over SSH (default), "
            "or query SLURM/LSF with --status/--runtime or --lsf-status/--lsf-runtime."
        )
    )
    parser.add_argument("id", help="Log/job id.")
    parser.add_argument(
        "--host",
        default=SSH_TARGET,
        help="SSH target in user@host format (default: %(default)s).",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Treat ID as a SLURM job id and print normalized status.",
    )
    mode_group.add_argument(
        "--runtime",
        action="store_true",
        help="Treat ID as a SLURM job id and print runtime (HH:MM:SS).",
    )
    mode_group.add_argument(
        "--lsf-status",
        action="store_true",
        help="Treat ID as an LSF job id and print normalized status.",
    )
    mode_group.add_argument(
        "--lsf-runtime",
        action="store_true",
        help="Treat ID as an LSF job id and print runtime (HH:MM:SS).",
    )
    args = parser.parse_args()

    try:
        if args.status:
            print_slurm_job_status(args.id, ssh_target=args.host)
            return 0
        if args.runtime:
            print_slurm_job_runtime(args.id, ssh_target=args.host)
            return 0
        if args.lsf_status:
            print_lsf_job_status(args.id, ssh_target=args.host)
            return 0
        if args.lsf_runtime:
            print_lsf_job_runtime(args.id, ssh_target=args.host)
            return 0

        content = load_remote_log(args.id, ssh_target=args.host)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
