# Sciantist

> **Autonomous AI Scientist** — A self-directed, closed-loop research system that ideates, implements, trains, evaluates, and iterates on machine learning experiments without human intervention - because not everyone can work 24/7.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Package manager: uv](https://img.shields.io/badge/package%20manager-uv-000000.svg)](https://github.com/astral-sh/uv)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [The Autonomous Loop](#the-autonomous-loop)
- [Worker Orchestration](#worker-orchestration)
- [Ideation System](#ideation-system)
- [Cluster Integration](#cluster-integration)
- [Experiment Tracking](#experiment-tracking)
- [Web UI](#web-ui)
- [MCP Tool Integration](#mcp-tool-integration)
- [Live Human-in-the-Loop Direction](#live-human-in-the-loop-direction)
- [Testing](#testing)
- [Extending Sciantist](#extending-sciantist)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Sciantist** is an autonomous AI-driven research orchestration framework that closes the full experimental loop for machine learning development. It continuously generates grounded experiment ideas, implements them via AI coding agents (Aider/OpenCode), submits training jobs to HPC clusters (SLURM/LSF), pulls metrics from Weights & Biases, and autonomously decides which changes to keep — building a leaderboard of the best-performing configurations over time.

Think of it as a **self-improving research pipeline**: given a target ML codebase, a training command, and a metric to optimize, Sciantist runs indefinitely, exploring the space of architectural, optimizer, hyperparameter, data augmentation, and loss function modifications — all without human intervention.

### Core Design Principles

| Principle | Description |
|-----------|-------------|
| **Closed-loop autonomy** | Ideate → Implement → Train → Evaluate → Decide → Repeat, fully automated |
| **Human-in-the-loop guidance** | Live direction via `user_prompt.md` — steer experiments in real time without stopping the loop |
| **Multi-agent parallelism** | Multiple independent workers explore different experiment branches simultaneously |
| **Expert-driven ideation** | Specialized AI personas (architecture, optimizer, loss, data, GPU, etc.) generate focused hypotheses |
| **Git-native isolation** | Every candidate runs in an isolated worktree on its own feature branch |
| **Resume-safe persistence** | Full checkpointing of state, leaderboard, and worker progress across restarts |
| **Cluster-agnostic** | Native support for SLURM and LSF schedulers via SSH to remote HPC systems |
| **Metric-driven decisions** | Configurable weighted metric composition with automatic keep/revert logic |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SCIANTIST ORCHESTRATOR                            │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Ideation │→ │  Aider/  │→ │  Git     │→ │ Cluster  │→ │  W&B Metric  │   │
│  │ Engine   │  │  OpenCode│  │ Worktree │  │ (SLURM)  │  │  Extraction  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│       ↑                                                       │             │
│       │                    ┌──────────────┐                   │             │
│       └────────────────────│  Keep/Revert │←──────────────────┘             │
│                            │  Decision    │                                 │
│                            └──────┬───────┘                                 │
│                                   │                                         │
│                            ┌──────▼───────┐                                 │
│                            │  Leaderboard │                                 │
│                            │  & State     │                                 │
│                            └──────────────┘                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Parallel Worker Pool                             │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │    │
│  │  │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker N │ │Expert W │  ...   │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Web UI (Flask Dashboard)                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Map

| Module | Responsibility |
|--------|---------------|
| `sciantist/ideation.py` | File discovery, prompt construction, MCP query parsing, expert payload generation |
| `sciantist/aider_ops.py` | Aider/OpenCode integration for planning and code implementation |
| `sciantist/repo_ops.py` | Git worktree creation, branch management, merge conflict resolution |
| `sciantist/cluster_ops.py` | SLURM/LSF job submission, status polling, remote log retrieval |
| `sciantist/wandb_ops.py` | W&B run snapshot extraction, metric history retrieval, retry logic |
| `sciantist/metrics.py` | Unified weighted metric computation, comparison logic |
| `sciantist/state.py` | Persistent state, leaderboard management, file locking, checkpointing |
| `sciantist/reporting.py` | Markdown/TSV experiment ledgers, LLM-run summaries |
| `sciantist/config.py` | Data models (`LoopConfig`, `ExperimentOutcome`, `StageOutcome`) |
| `sciantist/cli.py` | Full CLI with 40+ configuration flags |
| `sciantist/paths.py` | Deterministic path resolution for workers, worktrees, locks |
| `web_ui/web_ui.py` | Flask dashboard with leaderboard, worker status, run details |
| `llmclient/` | MCP-enabled LLM clients (MiniMax, web search, file reader, git, W&B) |

---

## Key Features

### 🧠 Autonomous Experiment Loop
- **Self-directed ideation** grounded in the actual codebase structure and prior results
- **Web-research pre-step** to incorporate the latest papers and techniques
- **LLM-powered keep/revert decisions** with unified metric comparison
- **Infinite loop mode** (`--no-forever` to disable) or single-iteration debugging

### 👥 Multi-Expert Ideation
Sciantist deploys **8 specialized expert personas** that generate focused experiment hypotheses:

| Expert | Focus Area |
|--------|-----------|
| **Hyperparameter Expert** | LR, warmup, scheduler, weight decay, dropout, batch size, EMA, clipping |
| **Architecture Expert** | Backbone swaps, depth/width, normalization, attention, adapters |
| **Loss Expert** | Weighted losses, auxiliary losses, focal variants, label smoothing, multi-task balancing |
| **Data Expert** | Preprocessing, augmentation, loading strategies, generalization improvements |
| **GPU Utilization Expert** | Throughput optimization, mixed precision, compilation, memory management |
| **Optimizer Expert** | AdamW/SGD/Lion alternatives, beta tuning, decoupled weight decay, LR schedules |
| **High-Risk/High-Reward Expert** | Novel, experimental ideas that challenge conventional assumptions |
| **Web-Research Expert** | Latest papers, preprints, conference proceedings, social media trends |

### ⚡ Parallel Worker Orchestration
- **Async worker pool** — configurable number of independent workers (`--async-worker-count`)
- **Expert worker pool** — dedicated workers per expert persona (`--expert-worker-count`)
- **Cross-process safety** — file-lock based synchronization for shared state
- **Stale detection** — automatic worker timeout and restart with exponential backoff
- **Heartbeat monitoring** — periodic health checks with configurable intervals

### 🔬 Git-Native Isolation
- **Worktree-per-candidate** — every experiment runs in an isolated git worktree
- **Feature branch naming** — deterministic `autoresearch/stage-N_c-M` convention
- **Merge conflict resolution** — Aider-assisted automatic conflict resolution
- **Automatic cleanup** — failed branches optionally removed (`--delete-failed-feature-branches`)
- **Push to origin** — all successful branches pushed for remote tracking

### 🖥️ HPC Cluster Integration
- **SLURM and LSF support** — normalized status mapping across schedulers
- **SSH-based remote execution** — submit jobs to any remote HPC cluster
- **Configurable runtime budgets** — HH:MM:SS format with automatic stop-before buffer
- **Remote log streaming** — pull stdout/stderr from remote machines
- **Cluster catalog YAML** — multi-cluster profiles with per-cluster settings

### 📊 Experiment Tracking & Metrics
- **Weights & Biases integration** — automatic run snapshot extraction with retry logic
- **Unified weighted metric** — configurable metric composition (e.g., `0.6 * accuracy + 0.4 * f1_score`)
- **Full metric histories** — per-step values retained for trend analysis
- **GPU utilization tracking** — average GPU utilization and memory percentage
- **Leaderboard persistence** — atomic JSON writes with "best in path" tracking
- **Markdown + TSV ledgers** — human-readable and machine-parseable experiment records

### 🌐 Web Dashboard
- **Flask-based UI** — real-time experiment monitoring
- **Leaderboard visualization** — ranked experiments with metric deltas
- **Worker status panel** — active/idle workers with heartbeat info
- **Run detail views** — full experiment outcomes with logs and summaries
- **JSON API** — RESTful endpoints for programmatic access

### 🔧 MCP (Model Context Protocol) Integration
- **Web Search MCP** — real-time research grounding for ideation
- **File Reader MCP** — secure file access for codebase analysis
- **Git MCP** — repository introspection and history queries
- **W&B MCP** — experiment data retrieval via MCP tool calls

### 🎯 Live Human-in-the-Loop Direction
- **`user_prompt.md`** — a living file that the system reads at every ideation step
- **Steer without stopping** — edit the file while the loop runs; new instructions take effect on the very next candidate
- **Priority guidance** — user instructions are injected as high-priority context, overriding default ideation when present
- **Use cases**: prioritize specific experiments, constrain the search space, react to intermediate results, enforce coding standards, or redirect focus mid-run
- **CLI override**: `--user-prompt-file /path/to/custom_prompt.md` to use a different file

---

## How It Works

### The Full Autonomous Cycle

```
1. IDEATE
   ├─ Scan codebase structure (files, summaries, prior results)
   ├─ Load expert personas and generate focused hypotheses
   ├─ Optional: web-research pre-step for latest paper integration
   ├─ Build structured ideation prompt with context
   └─ Parse LLM response into candidate experiment specs

2. IMPLEMENT
   ├─ Create isolated git worktree for candidate
   ├─ Launch Aider/OpenCode with plan + implementation prompts
   ├─ Auto-commit changes on feature branch
   └─ Push branch to origin

3. TRAIN
   ├─ Submit SLURM/LSF job with configured runtime budget
   ├─ Poll job status at configurable intervals
   ├─ Stream remote logs for monitoring
   └─ Detect terminal state (finished/crashed/timeout)

4. EVALUATE
   ├─ Pull W&B run snapshot (run_id == job_id)
   ├─ Extract metric histories and summary metrics
   ├─ Compute unified weighted metric
   ├─ Compare against baseline (higher/lower is better)
   └─ GPU utilization and memory analysis

5. DECIDE
   ├─ If metric improved → keep commit, merge to base branch
   ├─ If metric regressed → revert, discard feature branch
   ├─ If crashed → attempt Aider-assisted fix (up to N attempts)
   └─ Append outcome to leaderboard + experiments ledger

6. ITERATE
   ├─ Update stage memory with LLM summary
   ├─ Refresh file summary cache
   ├─ Check for new expert ideas
   └─ Loop back to step 1
```

### Decision Logic

```
                    ┌─────────────────┐
                    │  Job Finished?  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Pull W&B Data  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Compute Unified │
                    │    Metric       │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │  metric_new > metric_base?  │
              │  (or < if lower is better)  │
              └──────┬───────────────┬──────┘
                     │               │
                  YES│               │NO
                     │               │
              ┌──────▼──────┐  ┌─────▼──────┐
              │ KEEP change │  │ REVERT     │
              │ Merge to    │  │ Discard    │
              │ base branch │  │ branch     │
              └──────┬──────┘  └─────┬──────┘
                     │               │
              ┌──────▼───────────────▼──────┐
              │  Append to Leaderboard      │
              │  Update experiments.md/tsv  │
              │  Loop to next iteration     │
              └─────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** — fast Python package manager
- **Git** — for worktree and branch operations
- **SSH access** to a SLURM/LSF cluster (optional, for remote training)
- **Weights & Biases** account (optional, for experiment tracking)
- **LLM API access** — OpenAI-compatible endpoint (MiniMax, OpenRouter, etc.)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd sciantist

# Install dependencies with uv
uv sync

# Verify installation
uv run scian --help
```

### Basic Usage

```bash
# Run the autonomous loop with default configuration
uv run scian

# Run with a specific target repository
uv run scian --target-repo /path/to/ml-project \
             --train-command "./scripts/train.sh"

# Single iteration for debugging
uv run scian --single-iteration

# Dry run (no actual changes or job submissions)
uv run scian --dry-run
```

### Minimal Configuration

Create `config/.scian.yaml` in your project:

```yaml
target_repo: /path/to/your/ml-project
train_command: ./scripts/train.sh
model_name: gpt-4o
aider_model: openai/gpt-4o
openai_api_base: https://api.openai.com/
openai_api_key_env: OPENAI_API_KEY
wandb_project: your-wandb-project
metric_weights:
  val/loss: 1.0
metric_higher_is_better: false
```

---

## Configuration

### Configuration Priority

Settings are resolved in the following order (highest priority first):

1. **CLI arguments** — `--flag value`
2. **Repo config** — `config/.scian.yaml`
3. **Default config** — `config/default_config.yaml`
4. **Code constants** — hardcoded defaults in `sciantist/config.py`

### Key Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `target_repo` | `./target-project/` | Path to the ML codebase to experiment on |
| `train_command` | `./scripts/train.sh` | Command to start training |
| `model_name` | `gpt-4o` | LLM model for ideation and decision-making |
| `aider_model` | `openai/gpt-4o` | Model used by Aider for code implementation |
| `coder_backend` | `aider` | Code generation backend (`aider` or `opencode`) |
| `wandb_project` | `my-experiments` | W&B project for experiment tracking |
| `metric_weights` | `{"val/accuracy": 0.6, "val/f1_score": 0.4}` | Weighted metric composition |
| `metric_higher_is_better` | `true` | Direction of metric optimization |
| `async_worker_count` | `5` | Number of parallel workers |
| `expert_worker_count` | `5` | Number of expert-specific workers |
| `ideas_per_stage` | `5` | Candidates generated per stage |
| `max_fix_attempts` | `10` | Max Aider crash-fix attempts per candidate |
| `runtime` | `05:00:00` | Cluster job runtime budget (HH:MM:SS) |
| `poll_seconds` | `60` | Job polling interval in seconds |
| `stop_before_secs` | `660` | Buffer time before runtime deadline |
| `cluster_target` | `my-cluster` | Target cluster name |
| `output_dir` | `./outputs/sciantist/` | Directory for artifacts and state |

---

## CLI Reference

```
scian [OPTIONS]
```

### Core Options

| Flag | Type | Description |
|------|------|-------------|
| `--target-repo` | `str` | Path to the target ML repository |
| `--train-command` | `str` | Training command to execute on cluster |
| `--input-idea-path` | `str` | Path to initial idea/context file |
| `--user-prompt-file` | `str` | Custom user prompt for ideation |
| `--experts-path` | `str` | Path to expert persona definitions |
| `--branch` | `str` | Experiment branch name |
| `--output-dir` | `str` | Output directory for artifacts |

### LLM Configuration

| Flag | Type | Description |
|------|------|-------------|
| `--model-name` | `str` | LLM model for ideation |
| `--aider-model` | `str` | Model for Aider code generation |
| `--coder-backend` | `str` | Backend: `aider` or `opencode` |
| `--openai-api-base` | `str` | OpenAI-compatible API endpoint |
| `--openai-api-key-env` | `str` | Environment variable for API key |
| `--openai-api-key` | `str` | Direct API key (overrides env var) |

### Metric Configuration

| Flag | Type | Description |
|------|------|-------------|
| `--metric-weights` | `JSON` | Metric weights as JSON object |
| `--metric-higher-is-better` | `flag` | Higher metric values are better |
| `--metric-lower-is-better` | `flag` | Lower metric values are better |
| `--start-out-metric-baseline` | `float` | Baseline metric for comparison |

### Worker & Parallelism

| Flag | Type | Description |
|------|------|-------------|
| `--async-worker-mode` | `flag` | Enable async parallel workers |
| `--no-async-worker-mode` | `flag` | Disable async workers |
| `--async-worker-count` | `int` | Number of parallel workers |
| `--expert-worker-count` | `int` | Number of expert workers |
| `--worker-restart-backoff-seconds` | `int` | Backoff between worker restarts |
| `--worker-heartbeat-seconds` | `int` | Worker heartbeat interval |
| `--worker-stale-timeout-seconds` | `int` | Timeout before worker considered stale |

### Cluster Configuration

| Flag | Type | Description |
|------|------|-------------|
| `--cluster-target` | `str` | Target cluster name |
| `--cluster-extra-args` | `str` | Additional cluster submission args |
| `--cluster-config` | `str` | Path to cluster catalog YAML |
| `--cluster-name` | `str` | Specific cluster profile to use |
| `--runtime` | `str` | Job runtime (HH:MM:SS) |
| `--poll-seconds` | `int` | Job polling interval |
| `--stop-before-secs` | `int` | Buffer before runtime deadline |

### Execution Control

| Flag | Type | Description |
|------|------|-------------|
| `--dry-run` | `flag` | Simulate without making changes |
| `--single-iteration` | `flag` | Run exactly one iteration |
| `--no-forever` | `flag` | Stop after one full stage |
| `--max-fix-attempts` | `int` | Max crash fix attempts per candidate |
| `--max-wandb-retries` | `int` | Max W&B API retry attempts |
| `--verbose` | `flag` | Enable verbose logging |

### File & Repo Control

| Flag | Type | Description |
|------|------|-------------|
| `--no-worktrees` | `flag` | Disable git worktree isolation |
| `--allow-dirty` | `flag` | Allow uncommitted changes in repo |
| `--delete-failed-feature-branches` | `flag` | Remove branches for failed experiments |
| `--no-candidate-output-subdirs` | `flag` | Flatten output directory structure |
| `--no-file-summaries` | `flag` | Disable file summary caching |
| `--max-summary-files` | `int` | Max files to summarize |
| `--max-file-summary-chars` | `int` | Max chars per file summary |
| `--deny-pattern` | `list` | Glob patterns to exclude from editing |
| `--aider-only-pattern` | `list` | Patterns only editable by Aider |
| `--allowed-file-suffix` | `list` | Allowed file extensions |

### Ideation Control

| Flag | Type | Description |
|------|------|-------------|
| `--ideas-per-stage` | `int` | Number of candidates per stage |
| `--websearch-idea-prestep-enabled` | `flag` | Enable web research before ideation |
| `--no-websearch-idea-prestep-enabled` | `flag` | Disable web research pre-step |
| `--stage-multi-ideation-prompts` | `flag` | One prompt per candidate (vs batched) |
| `--no-stage-baseline` | `flag` | Skip baseline run at stage start |

---

## The Autonomous Loop

### Stage-Based Execution

Sciantist organizes experiments into **stages**, each consisting of multiple candidate experiments:

```
Stage 0
├── Baseline run (optional, --no-stage-baseline to skip)
├── Candidate 0: "Add gradient clipping at 1.0"
├── Candidate 1: "Switch to Lion optimizer with β1=0.95"
├── Candidate 2: "Increase batch size to 512 with gradient accumulation"
├── Candidate 3: "Add label smoothing ε=0.1"
├── Candidate 4: "Replace LayerNorm with RMSNorm"
└── Stage summary → merge best candidates → new baseline

Stage 1
├── New baseline (merged improvements from Stage 0)
├── Candidate 0: "Add cosine LR decay with warmup"
├── ...
```

### File Summary Cache

To avoid re-reading the entire codebase on every iteration, Sciantist maintains a **file summary cache** (`file_summary_cache.json`):

- Summarizes key files in the target repo using LLM
- Cached summaries are included in ideation prompts for context
- Configurable max files (`--max-summary-files`) and chars per summary (`--max-file-summary-chars`)
- Incremental updates when new files are added

### Memory System

Each output directory can contain a `memory.md` file that accumulates **project-level knowledge**:

- Prior experiment outcomes and lessons learned
- Failed approaches to avoid
- Promising directions to explore
- Updated automatically by LLM after each stage

---

## Worker Orchestration

### Async Worker Mode

When `--async-worker-mode` is enabled (default), Sciantist launches a pool of independent workers:

```
┌──────────────────────────────────────────────────────────────┐
│                      Orchestrator                            │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │  Worker 1   │  │  Worker 2   │  │  Worker 3   │  ...      │
│  │  (General)  │  │  (General)  │  │  (Expert)   │           │
│  │             │  │             │  │             │           │
│  │ ideate →    │  │ ideate →    │  │ ideate →    │           │
│  │ implement → │  │ implement → │  │ implement → │           │
│  │ train →     │  │ train →     │  │ train →     │           │
│  │ evaluate →  │  │ evaluate →  │  │ evaluate →  │           │
│  │ decide      │  │ decide      │  │ decide      │           │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
│         │                │                │                  │
│  ┌──────▼────────────────▼────────────────▼─────-─┐          │
│  │              Shared State                      │          │
│  │  • leaderboard.json (atomic writes)            │          │
│  │  • .sciantist.lock (cross-process lock)        │          │
│  │  • memory.md (project knowledge)               │          │
│  │  • experiments.md / .tsv (ledgers)             │          │
│  └────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

### Worker State Persistence

Each worker maintains persistent state for crash recovery:

```
outputs/sciantist/
├── workers/
│   ├── worker_01/
│   │   ├── worker_state.json      # Current run, stage, status
│   │   ├── worker_info.json       # Heartbeat, start time, config
│   │   ├── memory.md              # Worker-local memory
│   │   └── run_000001/
│   │       ├── run_state.json     # Trial checkpoint for resume
│   │       ├── run_info.json      # Run metadata
│   │       └── ...
│   └── worker_02/
│       └── ...
├── leaderboard.json               # Canonical experiment rankings
├── memory.md                      # Shared project memory
├── experiments.md                 # Human-readable ledger
├── experiments.tsv                # Machine-parseable ledger
└── sciantist.lock                 # Cross-process file lock
```

### Stale Detection & Recovery

Workers are monitored via heartbeat files:

- **Heartbeat interval**: `--worker-heartbeat-seconds` (default: 60s)
- **Stale timeout**: `--worker-stale-timeout-seconds` (default: 21600s = 6 hours)
- **Restart backoff**: `--worker-restart-backoff-seconds` (default: 5s)

If a worker's heartbeat exceeds the stale timeout, it is automatically restarted.

---

## Ideation System

### Prompt Construction Pipeline

```
1. Scan codebase
   ├─ List all files (respecting deny patterns)
   ├─ Read file summary cache
   ├─ Load expert persona definitions
   ├─ Load prior tried ideas (from leaderboard)
   └─ Read project memory (memory.md)

2. Build ideation prompt
   ├─ System context (experiment rules, constraints)
   ├─ Codebase glimpse (file listing)
   ├─ File summaries (LLM-generated summaries of key files)
   ├─ Expert instructions (persona-specific guidance)
   ├─ Tried ideas summary (what's been attempted, what worked)
   ├─ Prior experiment outcomes (experiments.md excerpts)
   └─ User prompt (custom instructions)

3. Optional: Web-research pre-step
   ├─ Query latest papers via web search MCP
   ├─ Format research findings for ideation context
   └─ Augment ideation prompt with latest research

4. Query LLM
   ├─ Send structured prompt via MiniMax/OpenRouter client
   ├─ Parse JSON response with candidate specs
   └─ Normalize into IdeaPayload objects

5. Execute candidates
   └─ For each candidate: implement → train → evaluate → decide
```

### Expert Personas

Experts are defined in `config/expert.md` as markdown sections. Each expert has:

- A **role title** (e.g., `# Architecture Expert`)
- A **system prompt** describing their expertise and constraints
- **Focus areas** specific to their domain
- **Risk tolerance** and **change scope** guidelines

New experts can be added by appending sections to the expert file.

---

## Cluster Integration

### Supported Schedulers

| Scheduler | Status Commands | Terminal States |
|-----------|----------------|-----------------|
| **SLURM** | `squeue`, `sacct` | COMPLETED → finished, FAILED/TIMEOUT/OOM → crashed |
| **LSF** | `bjobs` | DONE → finished, EXIT/ZOMBI → crashed |

### Cluster Profile Configuration

Define multiple clusters in `config/.scian-clusters.yaml`:

```yaml
hpc-cluster:
  cluster_target: hpc-cluster
  ssh_target: user@hpc.example.edu
  scheduler: slurm
  info_log_path: "$HOME/logs/{log_id}.out"
  error_log_path: "$HOME/logs/{log_id}.err"
  submit_extra_args: "--gres=gpu:4"
  wandb_sync: true

local:
  cluster_target: local
  ssh_target: localhost
  scheduler: slurm
  info_log_path: "$HOME/logs/{log_id}.out"
  error_log_path: "$HOME/logs/{log_id}.err"
  submit_extra_args: ""
  wandb_sync: false
```

### Job Submission Flow

```
1. Build training command
   ├─ Resolve train command path for target repo
   ├─ Inject W&B run_id (matches job_id)
   └─ Configure environment variables

2. Submit via cluster runner script
   ├─ Call run_on_cluster.sh with target, command, runtime
   └─ Parse job ID from output

3. Poll until terminal
   ├─ Query squeue/sacct at poll interval
   ├─ Track runtime against budget
   └─ Stop before deadline buffer

4. Retrieve results
   ├─ Pull remote logs via SSH
   ├─ Fetch W&B run snapshot
   └─ Extract metrics and GPU stats
```

---

## Experiment Tracking

### Weights & Biases Integration

Sciantist integrates with W&B for comprehensive experiment tracking:

- **Run ID correlation**: `run_id == job_id` for easy lookup
- **Automatic snapshot extraction**: pulls summary metrics, last-step metrics, and full histories
- **Retry logic**: exponential backoff for transient W&B API failures (up to `--max-wandb-retries`)
- **Metric history preservation**: full per-step values retained for trend analysis
- **GPU metrics**: average GPU utilization and memory percentage extracted from W&B

### Unified Metric Computation

```python
# Example: weighted metric composition
metric_weights = {
    "val/accuracy": 0.6,
    "val/f1_score": 0.4,
}

# Unified metric = weighted average of configured metrics
unified_metric = sum(metric_value * weight for metric, weight in weights.items()) / sum(weights.values())
```

### Leaderboard

The leaderboard (`leaderboard.json`) tracks all completed experiments:

```json
[
  {
    "timestamp_utc": "2026-05-19T10:30:00+00:00",
    "idea_title": "Add gradient clipping at 1.0",
    "feature_branch": "autoresearch/stage-0001_c00",
    "status": "finished",
    "unified_metric": 0.4521,
    "baseline_metric": 0.4380,
    "metric_delta": 0.0141,
    "kept": true,
    "currently_in_best_path": true,
    "runtime_seconds": 18000,
    "avg_gpu_util": 0.95,
    "summary": "Gradient clipping improved stability and validation accuracy by 1.4%..."
  }
]
```

---

## Web UI

Sciantist includes a **Flask-based web dashboard** for real-time experiment monitoring:

### Features

- **Leaderboard view**: ranked experiments with metric deltas and keep/revert status
- **Worker status panel**: active/idle workers with heartbeat and current run info
- **Run detail views**: full experiment outcomes with logs, summaries, and metric histories
- **JSON API**: RESTful endpoints for programmatic access

### Running the Dashboard

```bash
cd web_ui
uv run python web_ui.py --output-dir ../outputs/sciantist
```

The dashboard serves on `http://localhost:5000` by default.

---

## MCP Tool Integration

Sciantist leverages the **Model Context Protocol (MCP)** for tool-augmented LLM interactions:

### Available MCP Servers

| Server | Purpose | Implementation |
|--------|---------|---------------|
| **Web Search** | Real-time research grounding for ideation | `uvx web-search-mcp` |
| **File Reader** | Secure file access for codebase analysis | `llmclient/file_reader_mcp.py` |
| **Git** | Repository introspection and history | `llmclient/git_mcp.py` |
| **W&B** | Experiment data retrieval | `llmclient/wandb_mcp.py` |

### LLM Client

The `MiniMaxMCPClient` (`llmclient/minimax_client.py`) provides:

- OpenAI-compatible API interface
- Multi-tool call support with round-robin execution
- Think-block stripping for clean output
- Configurable API base (MiniMax, OpenRouter, local gateways)

---

## Live Human-in-the-Loop Direction

Sciantist supports **live, real-time steering** of the autonomous loop without requiring a restart or interruption.

### How It Works

At every ideation step, the system reads `config/user_prompt.md` (or a custom file via `--user-prompt-file`) and injects its contents as **high-priority context** into the ideation prompt. This means:

1. **Edit the file** while the loop is running
2. **New instructions take effect** on the very next candidate generation
3. **No restart needed** — the loop picks up your changes automatically

### Use Cases

| Scenario | Example `user_prompt.md` Content |
|----------|----------------------------------|
| **Prioritize a specific experiment** | "Focus on optimizer changes — try AdamW with decoupled weight decay." |
| **Constrain the search space** | "Only modify files in `src/training/`. Do not touch the data pipeline." |
| **React to intermediate results** | "The last 3 experiments with LR changes failed. Switch to architecture modifications." |
| **Enforce coding standards** | "All changes must pass `uv run pytest` and follow PEP 8. Use type hints everywhere." |
| **Redirect focus mid-run** | "We've explored hyperparameters enough. Now focus on data augmentation strategies." |
| **Seed a specific idea** | "Try using pretrained weights from `config.pretrained_checkpoint` and evaluate both model variants." |

### Priority System

The user prompt is injected as the **final section** of the ideation prompt, giving it maximum influence over the LLM's output. It overrides default ideation tendencies and expert suggestions when there's a conflict.

### Example Workflow

```bash
# 1. Start the autonomous loop
uv run scian

# 2. In another terminal, steer the experiments
echo "Focus on learning rate scheduling — try cosine annealing with warm restarts." > config/user_prompt.md

# 3. The next candidate will prioritize your instruction automatically

# 4. Change direction mid-run
echo "Switch to architecture experiments. Try adding a residual connection in the encoder." > config/user_prompt.md
```

### CLI Configuration

```bash
# Use a custom user prompt file
uv run scian --user-prompt-file /path/to/my_instructions.md

# Or configure it in config/.scian.yaml
# user_prompt_file: /path/to/my_instructions.md
```

---

## Testing

Sciantist includes a **comprehensive test suite** with 18 test modules:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=sciantist --cov=llmclient

# Run specific test module
uv run pytest tests/test_metrics.py -v
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `test_config.py` | LoopConfig defaults, validation, dataclass behavior |
| `test_metrics.py` | Unified metric computation, comparison logic, edge cases |
| `test_state.py` | State persistence, leaderboard operations, file locking |
| `test_ideation.py` | Prompt construction, JSON parsing, file discovery |
| `test_repo_ops.py` | Git operations, worktree management, merge resolution |
| `test_cluster_ops.py` | SLURM/LSF status mapping, cluster profiles |
| `test_wandb_ops.py` | W&B snapshot extraction, retry logic |
| `test_reporting.py` | Markdown/TSV rendering, metric history formatting |
| `test_paths.py` | Path resolution, worker directory structure |
| `test_minimax_client.py` | LLM client, tool call handling, response parsing |
| `test_file_reader_mcp.py` | File reader MCP server |
| `test_git_mcp.py` | Git MCP server |
| `test_wandb_mcp.py` | W&B MCP server |
| `test_aider_ops.py` | Aider integration helpers |
| `test_logging_utils.py` | Loguru configuration |
| `test_tracking_backend_integration.py` | End-to-end tracking tests |

---

## Extending Sciantist

### Adding a New Expert Persona

Append a new section to `config/expert.md`:

```markdown
# Regularization Expert
You are the regularization expert for iterative ML experiments.
Focus on techniques like L1/L2 penalties, dropout variants, stochastic depth,
and spectral normalization. Propose experiments with clear regularization
strengths and expected impact on generalization.
```

### Adding a Custom Metric

Update the `metric_weights` configuration:

```yaml
metric_weights:
  val/accuracy: 0.5
  val/f1_score: 0.3
  val/loss: 0.2
metric_higher_is_better: true
```

### Adding a New Cluster

Add a profile to `config/.scian-clusters.yaml`:

```yaml
my-cluster:
  cluster_target: my-cluster
  ssh_target: user@my-cluster.example.com
  scheduler: slurm
  info_log_path: "$HOME/logs/{log_id}.out"
  error_log_path: "$HOME/logs/{log_id}.err"
  submit_extra_args: "--partition=gpu --gres=gpu:8"
  wandb_sync: true
```

### Custom Ideation Prompts

Create a custom `user_prompt.md` and point to it:

```bash
uv run scian --user-prompt-file /path/to/custom_prompt.md
```

---

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install with dev dependencies
uv sync

# Run linter and formatter
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=sciantist --cov-report=html
```

---

## Roadmap

Planned features and integrations for upcoming releases:

| Status | Feature | Description |
|--------|---------|-------------|
| 🔜 Planned | **Trackio Integration** | Add [Trackio](https://github.com/gradio-app/trackio) as an alternative experiment tracking backend alongside W&B. Trackio provides lightweight, self-hosted experiment visualization with no external service dependency. Planned work includes: a `trackio_ops.py` module mirroring `wandb_ops.py`, a `--tracking-backend trackio` CLI flag, Trackio MCP server for experiment data retrieval, and unified metric extraction that works identically across both backends. |
| 🔜 Planned | **Multi-backend tracking** | Seamless switching between W&B and Trackio via `--tracking-backend <wandb|trackio>`, with a shared abstraction layer for run snapshots, metric histories, and GPU stats. |
| 🔜 Planned | **Trackio dashboard embedding** | Embed Trackio's native UI within the Sciantist Flask dashboard for unified experiment browsing. |

---

## License

This project is licensed under the Apache License, Version 2.0 — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **[Aider](https://aider.chat/)** — AI pair programming for code implementation
- **[Weights & Biases](https://wandb.ai/)** — Experiment tracking and visualization
- **[Model Context Protocol](https://modelcontextprotocol.io/)** — Standardized tool-use for LLMs
- **[SLURM](https://slurm.schedmd.com/)** — HPC job scheduling
- **[uv](https://github.com/astral-sh/uv)** — Fast Python package management
- **[MiniMax](https://www.minimax.io/)** — LLM API provider

---

> **Sciantist** — Because the best experiments are the ones you don't have to run manually.
