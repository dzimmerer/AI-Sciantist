"""Markdown/TSV reporting helpers for experiment outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from sciantist import ideation
from sciantist.config import ExperimentOutcome, LoopConfig


def _tail_text(text: str, max_chars: int = 8000) -> str:
    """Return trailing text for concise logging/prompts."""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _format_runtime_dd_hh_mm_ss(runtime_seconds: int | None) -> str:
    """Format runtime seconds as DD-HH:MM:SS."""
    if runtime_seconds is None:
        return "(unknown)"
    total_seconds = max(0, int(runtime_seconds))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}"


def _candidate_label_from_feature_branch(feature_branch: str) -> str:
    """Infer stage candidate label from feature branch naming."""
    if "stage-baseline" in feature_branch:
        return "baseline"
    marker = "_c"
    index = feature_branch.rfind(marker)
    if index == -1:
        return "unknown"

    digits: list[str] = []
    for char in feature_branch[index + len(marker) :]:
        if char.isdigit():
            digits.append(char)
            continue
        break
    if not digits:
        return "unknown"
    return f"candidate_{int(''.join(digits)):02d}"


def _summarize_metric_histories(histories: dict[str, list[float]] | None) -> str:
    """Return compact trend summary for metric histories."""
    if not isinstance(histories, dict) or not histories:
        return "(no metric histories)"

    rows: list[str] = []
    for metric_name, values in histories.items():
        if not isinstance(values, list) or not values:
            continue
        rows.append(
            (
                f"- {metric_name}: steps={len(values)}, first={values[0]:.6f}, "
                f"last={values[-1]:.6f}, best={max(values):.6f}"
            )
        )
    if not rows:
        return "(no metric histories)"
    return "\n".join(rows)


def _render_metric_histories_markdown_lines(histories: dict[str, list[float]] | None) -> list[str]:
    """Render detailed metric histories with summary stats and full per-step values."""
    lines = ["### Metric Histories"]
    if not isinstance(histories, dict) or not histories:
        lines.append("(no metric histories)")
        lines.append("")
        return lines

    has_any_values = False
    for metric_name in sorted(histories.keys()):
        values = histories.get(metric_name)
        if not isinstance(values, list) or not values:
            continue
        has_any_values = True
        values_rendered = ", ".join(f"{value:.6f}" for value in values)
        lines.extend(
            [
                f"- {metric_name}",
                f"  - steps: {len(values)}",
                f"  - first: {values[0]:.6f}",
                f"  - last: {values[-1]:.6f}",
                f"  - best: {max(values):.6f}",
                f"  - values_by_step: [{values_rendered}]",
            ]
        )

    if not has_any_values:
        lines.append("(no metric histories)")

    lines.append("")
    return lines


def _render_experiment_outcome_markdown(
    outcome: ExperimentOutcome,
    heading_level: int = 2,
    include_logs: bool = True,
) -> str:
    """Render one experiment outcome as markdown text."""
    heading_prefix = "#" * max(1, heading_level)
    lines = [
        f"{heading_prefix} {outcome.timestamp_utc} - {outcome.idea_title}",
        f"- candidate_label: {_candidate_label_from_feature_branch(outcome.feature_branch)}",
        f"- idea_branch_name: {outcome.idea_branch_name}",
        f"- feature_branch: {outcome.feature_branch}",
        f"- baseline_commit: {outcome.baseline_commit}",
        f"- trial_commit: {outcome.trial_commit}",
        f"- job_id: {outcome.job_id}",
        f"- wandb_project: {outcome.wandb_project}",
        f"- status: {outcome.status}",
        f"- runtime: {_format_runtime_dd_hh_mm_ss(outcome.runtime_seconds)}",
        f"- runtime_seconds: {outcome.runtime_seconds}",
        f"- unified_metric: {outcome.unified_metric}",
        f"- metric_histories: {json.dumps(outcome.metric_histories or {}, ensure_ascii=True)}",
        f"- baseline_metric: {outcome.baseline_metric}",
        f"- metric_delta: {outcome.metric_delta}",
        f"- avg_gpu_util: {outcome.avg_gpu_util}",
        f"- avg_gpu_memory_pct: {outcome.avg_gpu_memory}",
        f"- kept: {outcome.kept}",
        "",
    ]
    lines.extend(_render_metric_histories_markdown_lines(outcome.metric_histories))
    lines.extend(
        [
            "### Outline",
            outcome.idea_outline,
            "",
            "### Summary",
            outcome.summary,
            "",
        ]
    )
    if include_logs:
        lines.extend(
            [
                "### Error Log Excerpt",
                "```text",
                outcome.err_log_excerpt or "(no stderr excerpt)",
                "```",
                "",
                "### Info Log Excerpt",
                "```text",
                outcome.info_log_excerpt or "(no stdout excerpt)",
                "```",
                "",
                "### W&B Log Excerpt",
                "```text",
                outcome.wandb_log_excerpt or "(no wandb log excerpt)",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _write_stage_candidate_summary_markdown(stage_dir: Path, label: str, outcome: ExperimentOutcome) -> Path:
    """Write one candidate summary markdown in its stage candidate folder."""
    candidate_dir = stage_dir / "candidates" / label
    candidate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = candidate_dir / "summary.md"
    summary_path.write_text(
        _render_experiment_outcome_markdown(outcome, heading_level=2, include_logs=False), encoding="utf-8"
    )
    return summary_path


def _write_stage_summary_markdown(
    stage_dir: Path,
    stage_index: int,
    stage_summary_text: str,
    stage_baseline_metric: float | None,
    merged_feature_branches: list[str],
    skipped_conflict_feature_branches: list[str],
    reset_to_best: bool,
    stage_improvement_ideas: str,
    stage_outcomes: list[ExperimentOutcome],
) -> Path:
    """Write a stage-level markdown summary with synthesized improvement ideas."""
    rendered_outcomes = "\n\n".join(
        _render_experiment_outcome_markdown(outcome, heading_level=3, include_logs=False) for outcome in stage_outcomes
    )
    lines = [
        f"# Stage {stage_index:04d} Summary",
        "",
        "## Stage Decision Summary",
        stage_summary_text,
        "",
        "## Stage Metrics and Merge Outcomes",
        f"- stage_baseline_metric: {stage_baseline_metric}",
        f"- merged_feature_branches: {merged_feature_branches}",
        f"- skipped_conflict_feature_branches: {skipped_conflict_feature_branches}",
        f"- reset_to_best: {reset_to_best}",
        "",
        "## Improvement Ideas (LLM + Websearch)",
        "```json",
        stage_improvement_ideas,
        "```",
        "",
        "## Candidate Summaries",
        rendered_outcomes,
        "",
    ]
    summary_path = stage_dir / "stage_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def _summarize_run_with_llm(
    config: LoopConfig,
    outcome: ExperimentOutcome,
    diff_patch: str,
    decision_summary: str,
) -> str:
    """Generate a per-run narrative summary from plan, changes, and metrics."""
    try:
        client = ideation._make_minimax_client(config)
        metric_histories_json = _tail_text(
            json.dumps(outcome.metric_histories or {}, ensure_ascii=True),
            max_chars=8000,
        )
        prompt = (
            "Write a concise but specific per-run report for an autonomous experiment. "
            "You must include exact changed values or parameter names when visible in the diff. "
            "If values are not explicit, say so. Use clear sections and short bullets.\n\n"
            f"Run metadata:\n"
            f"- idea_title: {outcome.idea_title}\n"
            f"- idea_branch_name: {outcome.idea_branch_name}\n"
            f"- feature_branch: {outcome.feature_branch}\n"
            f"- baseline_commit: {outcome.baseline_commit}\n"
            f"- trial_commit: {outcome.trial_commit}\n"
            f"- job_id: {outcome.job_id}\n"
            f"- wandb_project: {outcome.wandb_project}\n"
            f"- wandb_run_id: {outcome.job_id}\n"
            f"- status: {outcome.status}\n"
            f"- runtime_seconds: {outcome.runtime_seconds}\n"
            f"- unified_metric: {outcome.unified_metric}\n"
            f"- metric_histories: {metric_histories_json}\n"
            f"- metric_history_trends:\n{_summarize_metric_histories(outcome.metric_histories)}\n"
            f"- baseline_metric: {outcome.baseline_metric}\n"
            f"- metric_delta: {outcome.metric_delta}\n"
            f"- avg_gpu_util: {outcome.avg_gpu_util}\n"
            f"- avg_gpu_memory: {outcome.avg_gpu_memory}\n"
            f"- kept: {outcome.kept}\n"
            f"- decision_summary: {decision_summary}\n\n"
            f"Original planning prompt:\n{outcome.aider_plan_prompt or '(none)'}\n\n"
            f"Original implementation prompt:\n{outcome.aider_impl_prompt or '(none)'}\n\n"
            f"Diff patch (unified=0):\n{diff_patch}\n\n"
            f"Info log excerpt:\n{outcome.info_log_excerpt or '(no stdout excerpt)'}\n\n"
            f"Error log excerpt:\n{outcome.err_log_excerpt or '(no stderr excerpt)'}\n\n"
            "Required output format:\n"
            "1) What changed\n"
            "2) Exact parameter/value edits\n"
            "3) Result metrics and system utilization\n"
            "4) Outcome assessment\n"
            "Keep it factual and grounded only in provided data."
        )
        response = client.make_query_with_wandb(
            query=prompt,
            system_prompt=(
                "You are a precise ML experiment reporter. Do not hallucinate values; mark unknowns explicitly."
            ),
        )
        rendered = response.strip()
        return _tail_text(rendered, max_chars=7000) if rendered else decision_summary
    except Exception as error:
        return f"{decision_summary} LLM per-run summarization failed: {error}"


def _append_experiments_md(path: Path, outcome: ExperimentOutcome) -> None:
    """Append a structured markdown entry for one experiment."""
    lines = [
        f"## {outcome.timestamp_utc} - {outcome.idea_title}",
        f"- idea_branch_name: {outcome.idea_branch_name}",
        f"- feature_branch: {outcome.feature_branch}",
        f"- baseline_commit: {outcome.baseline_commit}",
        f"- trial_commit: {outcome.trial_commit}",
        f"- job_id: {outcome.job_id}",
        f"- wandb_project: {outcome.wandb_project}",
        f"- status: {outcome.status}",
        f"- runtime_seconds: {outcome.runtime_seconds}",
        f"- unified_metric: {outcome.unified_metric}",
        f"- metric_histories: {json.dumps(outcome.metric_histories or {}, ensure_ascii=True)}",
        f"- baseline_metric: {outcome.baseline_metric}",
        f"- metric_delta: {outcome.metric_delta}",
        f"- avg_gpu_util: {outcome.avg_gpu_util}",
        f"- avg_gpu_memory: {outcome.avg_gpu_memory}",
        f"- kept: {outcome.kept}",
        "",
        "### Outline",
        outcome.idea_outline,
        "",
        "### Summary",
        outcome.summary,
        "",
        # "### Error Log Excerpt",
        # "```text",
        # outcome.err_log_excerpt or "(no stderr excerpt)",
        # "```",
        # "",
        # "### Info Log Excerpt",
        # "```text",
        # outcome.info_log_excerpt or "(no stdout excerpt)",
        # "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))


def _append_experiments_tsv(path: Path, outcome: ExperimentOutcome) -> None:
    """Append a TSV row for quick tabular analysis."""
    header = (
        "timestamp_utc\tidea_title\tidea_branch_name\tfeature_branch\tbaseline_commit\ttrial_commit\tjob_id\twandb_project\tstatus\truntime_seconds"
        "\tunified_metric\tmetric_histories\tbaseline_metric\tmetric_delta\tavg_gpu_util\tavg_gpu_memory\tkept\tsummary\n"
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header, encoding="utf-8")

    summary_clean = outcome.summary.replace("\t", " ").replace("\n", " ")
    histories_json = json.dumps(outcome.metric_histories or {}, ensure_ascii=True, separators=(",", ":"))
    row = (
        f"{outcome.timestamp_utc}\t{outcome.idea_title}\t{outcome.idea_branch_name}\t{outcome.feature_branch}\t"
        f"{outcome.baseline_commit}\t{outcome.trial_commit}\t"
        f"{outcome.job_id}\t{outcome.wandb_project}\t{outcome.status}\t{outcome.runtime_seconds}\t{outcome.unified_metric}\t{histories_json}\t"
        f"{outcome.baseline_metric}\t{outcome.metric_delta}\t{outcome.avg_gpu_util}\t{outcome.avg_gpu_memory}\t"
        f"{outcome.kept}\t{summary_clean}\n"
    )
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(row)


def _is_outcome_already_logged(path: Path, outcome: ExperimentOutcome) -> bool:
    """Return True when an outcome is already present in the TSV ledger."""
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                if line.startswith("timestamp_utc\t"):
                    continue
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 8:
                    continue
                feature_branch = columns[3]
                baseline_commit = columns[4]
                trial_commit = columns[5]
                job_id = columns[6]
                if (
                    feature_branch == outcome.feature_branch
                    and baseline_commit == outcome.baseline_commit
                    and trial_commit == outcome.trial_commit
                    and job_id == outcome.job_id
                ):
                    return True
    except OSError:
        return False
    return False
