const API_BASE = '/api';

let currentExpPage = 1;
let currentWorkerId = null;
let currentRunId = null;
let currentExpSort = 'timestamp_utc';
let currentExpOrder = 'desc';

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initLeaderboardControls();
    initExperimentControls();
    initWorkerControls();
    initModal();
    loadOverview();
});

function initNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.dataset.section;
            showSection(section);
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    document.getElementById('back-to-workers')?.addEventListener('click', () => {
        showSection('workers');
    });

    document.getElementById('back-to-list')?.addEventListener('click', () => {
        if (currentWorkerId) {
            showWorkerDetail(currentWorkerId);
        } else {
            showSection('experiments');
        }
    });
    
    document.getElementById('refresh-output-log')?.addEventListener('click', loadGlobalLog);
}

function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    const section = document.getElementById(sectionId);
    if (section) {
        section.classList.add('active');
    }
    if (sectionId === 'output') {
        loadGlobalLog();
    }
}

async function loadGlobalLog() {
    const container = document.getElementById('global-output-content');
    const refreshBtn = document.getElementById('refresh-output-log');
    let currentOffset = 0;
    let isLoading = false;
    let hasMore = true;
    let isAtBottom = false;
    
    container.innerHTML = '<p class="loading">Loading...</p>';
    refreshBtn.disabled = true;
    
    async function loadChunk(offset) {
        if (isLoading || (!hasMore && offset > 0)) return;
        isLoading = true;
        try {
            const data = await apiFetch(`/global-log?limit=1000&offset=${offset}`);
            if (data.error) {
                container.innerHTML = `<p class="loading">${escapeHtml(data.error)}</p>`;
            } else {
                if (offset === 0) {
                    container.textContent = '';
                }
                const pre = document.createElement('pre');
                pre.style.whiteSpace = 'pre-wrap';
                pre.textContent = data.log;
                container.appendChild(pre);
                currentOffset = offset + 1000;
                hasMore = data.has_more;
                if (!hasMore) {
                    const endMarker = document.createElement('p');
                    endMarker.className = 'loading';
                    endMarker.textContent = '--- End of log ---';
                    container.appendChild(endMarker);
                }
            }
        } catch (error) {
            console.error('Failed to load global log:', error);
        } finally {
            isLoading = false;
            refreshBtn.disabled = false;
        }
    }
    
    container.addEventListener('scroll', () => {
        if (isAtBottom) return;
        const atBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;
        if (atBottom && hasMore && !isLoading) {
            isAtBottom = true;
            const loader = document.createElement('p');
            loader.className = 'loading';
            loader.textContent = 'Loading more...';
            container.appendChild(loader);
            loadChunk(currentOffset).then(() => {
                container.removeChild(loader);
                isAtBottom = false;
            });
        }
    });
    
    await loadChunk(0);
    
    refreshBtn.addEventListener('click', () => {
        currentOffset = 0;
        hasMore = true;
        container.innerHTML = '<p class="loading">Loading...</p>';
        refreshBtn.disabled = true;
        loadChunk(0);
    });
}

async function apiFetch(endpoint) {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    return response.json();
}

async function loadOverview() {
    try {
        const data = await apiFetch('/overview');
        updateOverviewStats(data.summary);
        updateRecentRuns(data.recent_runs);
        updateTopPerformers(data.best_runs);
        updateProjectMemory(data.memory);
    } catch (error) {
        console.error('Failed to load overview:', error);
        document.getElementById('stats-grid').innerHTML = `<p class="loading" style="color: var(--danger);">Error: ${error.message}</p>`;
    }
}

function updateOverviewStats(summary) {
    document.getElementById('stat-total-runs').textContent = summary.total_runs || 0;
    document.getElementById('stat-finished').textContent = summary.finished_runs || 0;
    document.getElementById('stat-failed').textContent = summary.failed_runs || 0;
    document.getElementById('stat-running').textContent = summary.in_progress_runs || 0;
    document.getElementById('stat-best-metric').textContent = summary.best_metric 
        ? summary.best_metric.toFixed(4) 
        : '-';
    document.getElementById('stat-avg-metric').textContent = summary.avg_metric 
        ? summary.avg_metric.toFixed(4) 
        : '-';
    document.getElementById('stat-workers').textContent = summary.unique_workers || 0;
}

function updateRecentRuns(runs) {
    const container = document.getElementById('recent-runs');
    if (!runs || runs.length === 0) {
        container.innerHTML = '<p class="loading">No recent runs</p>';
        return;
    }
    container.innerHTML = runs.map(run => createRunItem(run)).join('');
    container.querySelectorAll('.run-item').forEach(item => {
        item.addEventListener('click', () => showRunDetail(item.dataset.runId));
    });
}

function updateTopPerformers(runs) {
    const container = document.getElementById('top-performers');
    if (!runs || runs.length === 0) {
        container.innerHTML = '<p class="loading">No runs yet</p>';
        return;
    }
    container.innerHTML = runs.map(run => createRunItem(run)).join('');
    container.querySelectorAll('.run-item').forEach(item => {
        item.addEventListener('click', () => showRunDetail(item.dataset.runId));
    });
}

function createRunItem(run) {
    const statusClass = run.status === 'finished' ? 'status-finished' : 
                        run.status === 'crashed' ? 'status-crashed' : 'status-running';
    return `
        <div class="run-item" data-run-id="${escapeHtml(run.run_id)}">
            <div class="run-item-header">
                <span class="run-item-title">${escapeHtml(run.idea_title || 'Untitled')}</span>
                <span class="run-item-status ${statusClass}">${escapeHtml(run.status)}</span>
            </div>
            <div class="run-item-meta">
                <span>Worker: ${escapeHtml(run.worker_id || 'unknown')}</span>
                <span>Metric: <span class="run-item-metric">${run.unified_metric ? run.unified_metric.toFixed(4) : '-'}</span></span>
                <span>${formatTimestamp(run.timestamp_utc)}</span>
            </div>
        </div>
    `;
}

function updateProjectMemory(memory) {
    const container = document.getElementById('project-memory');
    const expandBtn = document.getElementById('expand-memory');
    if (!memory) {
        container.innerHTML = '<p class="loading">No memory available</p>';
        return;
    }
    container.textContent = memory;
    
    expandBtn?.addEventListener('click', () => {
        container.classList.toggle('expanded');
        if (container.classList.contains('expanded')) {
            expandBtn.textContent = 'Collapse';
        } else {
            expandBtn.textContent = 'Expand';
        }
    });
}

async function loadLeaderboard() {
    const sortBy = document.getElementById('leaderboard-sort').value;
    const order = document.getElementById('leaderboard-order').value;
    
    try {
        const data = await apiFetch(`/leaderboard?sort=${sortBy}&order=${order}`);
        updateLeaderboardTable(data.leaderboard);
    } catch (error) {
        console.error('Failed to load leaderboard:', error);
    }
}

function updateLeaderboardTable(runs) {
    const tbody = document.getElementById('leaderboard-tbody');
    if (!runs || runs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="loading">No runs in leaderboard</td></tr>';
        return;
    }
    
    tbody.innerHTML = runs.map((run, index) => `
        <tr>
            <td>${index + 1}</td>
            <td>${escapeHtml(run.worker_id || '-')}</td>
            <td><code>${escapeHtml(run.run_id || '-')}</code></td>
            <td>${escapeHtml(run.idea_title || '-')}</td>
            <td><span class="badge ${run.status === 'finished' ? 'badge-success' : run.status === 'crashed' ? 'badge-danger' : run.status === 'running' ? 'badge-info' : 'badge-warning'}">${escapeHtml(run.status)}</span></td>
            <td>${run.unified_metric ? run.unified_metric.toFixed(4) : '-'}</td>
            <td>${formatRuntime(run.runtime_seconds)}</td>
            <td>${formatTimestamp(run.timestamp_utc)}</td>
            <td><button class="action-btn" onclick="showRunDetail('${escapeHtml(run.run_id)}')">View</button></td>
        </tr>
    `).join('');
}

function initLeaderboardControls() {
    document.getElementById('refresh-leaderboard')?.addEventListener('click', loadLeaderboard);
    document.getElementById('leaderboard-sort')?.addEventListener('change', loadLeaderboard);
    document.getElementById('leaderboard-order')?.addEventListener('change', loadLeaderboard);
    
    document.querySelector('.nav-link[data-section="leaderboard"]')?.addEventListener('click', loadLeaderboard);
}

async function loadExperiments(page = 1) {
    const statusFilter = document.getElementById('exp-status-filter')?.value || 'all';
    const workerFilter = document.getElementById('exp-worker-filter')?.value || '';
    
    currentExpPage = page;
    
    try {
        const data = await apiFetch(`/experiments?page=${page}&status=${statusFilter}&worker=${encodeURIComponent(workerFilter)}&sort=${currentExpSort}&order=${currentExpOrder}`);
        updateExperimentsTable(data.experiments);
        updateExpPagination(data);
    } catch (error) {
        console.error('Failed to load experiments:', error);
    }
}

function updateExperimentsTable(experiments) {
    const tbody = document.getElementById('experiments-tbody');
    if (!experiments || experiments.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="loading">No experiments found</td></tr>';
        return;
    }
    
    tbody.innerHTML = experiments.map(run => `
        <tr>
            <td><code>${escapeHtml(run.run_id || '-')}</code></td>
            <td>${escapeHtml(run.worker_id || '-')}</td>
            <td>${escapeHtml(run.idea_title || '-')}</td>
            <td><span class="badge ${run.status === 'finished' ? 'badge-success' : run.status === 'crashed' ? 'badge-danger' : run.status === 'running' ? 'badge-info' : 'badge-warning'}">${escapeHtml(run.status)}</span></td>
            <td>${run.unified_metric ? run.unified_metric.toFixed(4) : '-'}</td>
            <td>${run.metric_delta ? (run.metric_delta > 0 ? '+' : '') + run.metric_delta.toFixed(4) : '-'}</td>
            <td>${formatRuntime(run.runtime_seconds)}</td>
            <td><code>${escapeHtml(run.job_id || '-')}</code></td>
            <td>${run.kept ? '<span class="badge badge-success">Yes</span>' : '<span class="badge badge-warning">No</span>'}</td>
            <td>${formatTimestamp(run.timestamp_utc)}</td>
            <td><button class="action-btn" onclick="showRunDetail('${escapeHtml(run.run_id)}')">View</button></td>
        </tr>
    `).join('');
}

function updateSortIndicators() {
    document.querySelectorAll('#experiments-table th[data-sort]').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        if (th.dataset.sort === currentExpSort) {
            th.classList.add(currentExpOrder === 'asc' ? 'sort-asc' : 'sort-desc');
        }
    });
}

function initExperimentControls() {
    document.getElementById('filter-experiments')?.addEventListener('click', () => loadExperiments(1));
    document.getElementById('exp-prev')?.addEventListener('click', () => loadExperiments(currentExpPage - 1));
    document.getElementById('exp-next')?.addEventListener('click', () => loadExperiments(currentExpPage + 1));
    
    document.querySelector('.nav-link[data-section="experiments"]')?.addEventListener('click', () => {
        loadExperiments(1).then(() => initSortHeaders());
    });
    
    loadExperiments(1).then(() => initSortHeaders());
}

function initSortHeaders() {
    const sortHeaders = document.querySelectorAll('#experiments-table th[data-sort]');
    sortHeaders.forEach(th => {
        th.addEventListener('click', () => {
            const sortKey = th.dataset.sort;
            if (currentExpSort === sortKey) {
                currentExpOrder = currentExpOrder === 'asc' ? 'desc' : 'asc';
            } else {
                currentExpSort = sortKey;
                currentExpOrder = 'desc';
            }
            loadExperiments(1);
        });
        th.style.cursor = 'pointer';
    });
}

function updateExperimentsTable(experiments) {
    const tbody = document.getElementById('experiments-tbody');
    if (!experiments || experiments.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="loading">No experiments found</td></tr>';
        return;
    }
    
    tbody.innerHTML = experiments.map(run => `
        <tr>
            <td><code>${escapeHtml(run.run_id || '-')}</code></td>
            <td>${escapeHtml(run.worker_id || '-')}</td>
            <td>${escapeHtml(run.idea_title || '-')}</td>
            <td><span class="badge ${run.status === 'finished' ? 'badge-success' : run.status === 'crashed' ? 'badge-danger' : run.status === 'running' ? 'badge-info' : 'badge-warning'}">${escapeHtml(run.status)}</span></td>
            <td>${run.unified_metric ? run.unified_metric.toFixed(4) : '-'}</td>
            <td>${run.metric_delta ? (run.metric_delta > 0 ? '+' : '') + run.metric_delta.toFixed(4) : '-'}</td>
            <td>${formatRuntime(run.runtime_seconds)}</td>
            <td><code>${escapeHtml(run.job_id || '-')}</code></td>
            <td>${run.kept ? '<span class="badge badge-success">Yes</span>' : '<span class="badge badge-warning">No</span>'}</td>
            <td>${formatTimestamp(run.timestamp_utc)}</td>
            <td><button class="action-btn" onclick="showRunDetail('${escapeHtml(run.run_id)}')">View</button></td>
        </tr>
    `).join('');
    
    updateSortIndicators();
}

function updateExpPagination(data) {
    const prevBtn = document.getElementById('exp-prev');
    const nextBtn = document.getElementById('exp-next');
    const pageInfo = document.getElementById('exp-page-info');
    
    if (prevBtn) prevBtn.disabled = currentExpPage <= 1;
    if (nextBtn) nextBtn.disabled = currentExpPage >= data.total_pages;
    if (pageInfo) pageInfo.textContent = `Page ${data.page} of ${data.total_pages || 1}`;
}

function initExperimentControls() {
    document.getElementById('filter-experiments')?.addEventListener('click', () => loadExperiments(1));
    document.getElementById('exp-prev')?.addEventListener('click', () => loadExperiments(currentExpPage - 1));
    document.getElementById('exp-next')?.addEventListener('click', () => loadExperiments(currentExpPage + 1));
    
    document.querySelector('.nav-link[data-section="experiments"]')?.addEventListener('click', () => {
        loadExperiments(1).then(() => initSortHeaders());
    });
    
    loadExperiments(1).then(() => initSortHeaders());
}

function initSortHeaders() {
    const sortHeaders = document.querySelectorAll('#experiments-table th[data-sort]');
    sortHeaders.forEach(th => {
        th.addEventListener('click', () => {
            const sortKey = th.dataset.sort;
            if (currentExpSort === sortKey) {
                currentExpOrder = currentExpOrder === 'asc' ? 'desc' : 'asc';
            } else {
                currentExpSort = sortKey;
                currentExpOrder = 'desc';
            }
            loadExperiments(1);
        });
        th.style.cursor = 'pointer';
    });
}

async function loadWorkers() {
    try {
        const data = await apiFetch('/workers');
        updateWorkersGrid(data.workers);
    } catch (error) {
        console.error('Failed to load workers:', error);
    }
}

function updateWorkersGrid(workers) {
    const grid = document.getElementById('workers-grid');
    if (!workers || workers.length === 0) {
        grid.innerHTML = '<p class="loading">No workers found</p>';
        return;
    }
    
    grid.innerHTML = workers.map(worker => {
        const stageClass = `stage-${worker.worker_stage || 'idle'}`;
        return `
            <div class="worker-card" onclick="showWorkerDetail('${escapeHtml(worker.worker_id)}')">
                <div class="worker-card-top">
                    <span class="worker-role-badge">${escapeHtml(worker.role)}</span>
                    <span class="worker-id-centered">${escapeHtml(worker.worker_id)}</span>
                </div>
                <div class="worker-stats">
                    <div class="worker-stat">
                        <div class="worker-stat-value">${worker.total_runs || 0}</div>
                        <div class="worker-stat-label">Total Runs</div>
                    </div>
                    <div class="worker-stat">
                        <div class="worker-stat-value">${worker.best_metric ? worker.best_metric.toFixed(4) : '-'}</div>
                        <div class="worker-stat-label">Best Metric</div>
                    </div>
                </div>
                <div class="worker-stage ${stageClass}">${escapeHtml(worker.worker_stage || 'idle')}</div>
            </div>
        `;
    }).join('');
}

function initWorkerControls() {
    document.querySelector('.nav-link[data-section="workers"]')?.addEventListener('click', loadWorkers);
    document.getElementById('load-worker-log-btn')?.addEventListener('click', loadWorkerLog);
}

async function showWorkerDetail(workerId) {
    currentWorkerId = workerId;
    document.getElementById('worker-detail-title').textContent = `Worker: ${workerId}`;
    showSection('worker-detail');
    
    document.getElementById('worker-log-content').style.display = 'none';
    document.getElementById('worker-log-content').textContent = '';
    document.getElementById('load-worker-log-btn').style.display = 'inline-block';
    
    try {
        const data = await apiFetch(`/worker/${encodeURIComponent(workerId)}`);
        updateWorkerDetail(data);
    } catch (error) {
        console.error('Failed to load worker detail:', error);
    }
}

async function loadWorkerLog() {
    if (!currentWorkerId) return;
    
    const btn = document.getElementById('load-worker-log-btn');
    const content = document.getElementById('worker-log-content');
    
    btn.disabled = true;
    btn.textContent = 'Loading...';
    
    try {
        const data = await apiFetch(`/worker/${encodeURIComponent(currentWorkerId)}/log`);
        content.textContent = data.log || 'No log available';
        content.style.display = 'block';
        btn.style.display = 'none';
    } catch (error) {
        console.error('Failed to load worker log:', error);
        content.textContent = 'Error loading log: ' + error.message;
        content.style.display = 'block';
    } finally {
        btn.disabled = false;
    }
}

function updateWorkerDetail(data) {
    const statusContent = document.getElementById('worker-status');
    const state = data.state || {};
    const info = data.info || {};
    
    statusContent.innerHTML = `
        <div class="status-row">
            <span class="status-row-label">Worker Stage</span>
            <span class="status-row-value">${escapeHtml(state.worker_stage || 'unknown')}</span>
        </div>
        <div class="status-row">
            <span class="status-row-label">Run Index</span>
            <span class="status-row-value">${state.run_index || 0}</span>
        </div>
        <div class="status-row">
            <span class="status-row-label">Active Run ID</span>
            <span class="status-row-value"><code>${escapeHtml(state.active_run_id || '-')}</code></span>
        </div>
        <div class="status-row">
            <span class="status-row-label">Last Heartbeat</span>
            <span class="status-row-value">${formatTimestamp(state.heartbeat_ts_utc)}</span>
        </div>
        <div class="status-row">
            <span class="status-row-label">Last Cycle</span>
            <span class="status-row-value">${formatTimestamp(state.last_cycle_ts_utc)}</span>
        </div>
        <div class="status-row">
            <span class="status-row-label">Heartbeat Phase</span>
            <span class="status-row-value">${escapeHtml(state.heartbeat_phase || '-')}</span>
        </div>
        ${state.error ? `
        <div class="status-row">
            <span class="status-row-label">Last Error</span>
            <span class="status-row-value">${formatTimestamp(state.last_cycle_error_ts_utc)}</span>
        </div>
        ` : ''}
    `;
    
    const infoContent = document.getElementById('worker-info');
    const expertContent = info.expert 
        ? `<div class="expert-name">${escapeHtml(info.expert.name || 'Expert')}</div><div class="expert-description">${escapeHtml(info.expert.description || '')}</div>`
        : '-';
    infoContent.innerHTML = `
        <div class="status-row">
            <span class="status-row-label">Expert</span>
            <span class="status-row-value expert-aligned">${expertContent}</span>
        </div>
        <div class="status-row">
            <span class="status-row-label">Current Run Index</span>
            <span class="status-row-value">${info.current_run_index ?? '-'}</span>
        </div>
        <div class="status-row">
            <span class="status-row-label">Last Updated</span>
            <span class="status-row-value">${formatTimestamp(info.updated_ts_utc)}</span>
        </div>
    `;
    
    const memoryContent = document.getElementById('worker-memory');
    memoryContent.textContent = data.memory || 'No worker memory available';
    
    updateWorkerRunsTable(data.runs);
}

function updateWorkerRunsTable(runs) {
    const tbody = document.getElementById('worker-runs-tbody');
    if (!runs || runs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="loading">No runs for this worker</td></tr>';
        return;
    }
    
    tbody.innerHTML = runs.map(run => `
        <tr>
            <td><code>${escapeHtml(run.run_id || '-')}</code></td>
            <td>${escapeHtml(run.idea_title || '-')}</td>
            <td><span class="badge ${run.status === 'finished' ? 'badge-success' : run.status === 'crashed' ? 'badge-danger' : run.status === 'running' ? 'badge-info' : 'badge-warning'}">${escapeHtml(run.status)}</span></td>
            <td>${run.unified_metric ? run.unified_metric.toFixed(4) : '-'}</td>
            <td>${formatRuntime(run.runtime_seconds)}</td>
            <td>${formatTimestamp(run.timestamp_utc)}</td>
            <td><button class="action-btn" onclick="showRunDetail('${escapeHtml(run.run_id)}')">View</button></td>
        </tr>
    `).join('');
}

async function showRunDetail(runId) {
    currentRunId = runId;
    showSection('run-detail');
    document.getElementById('run-detail-title').textContent = `Run: ${runId}`;
    
    try {
        const data = await apiFetch(`/run/${encodeURIComponent(runId)}`);
        updateRunDetail(data.run, data.parent_history || []);
    } catch (error) {
        console.error('Failed to load run detail:', error);
        document.getElementById('run-detail-content').innerHTML = '<p class="loading">Failed to load run details</p>';
    }
}

function updateRunDetail(run, parentHistory = []) {
    if (!run) {
        document.getElementById('run-detail-content').innerHTML = '<p class="loading">Run not found</p>';
        return;
    }
    
    const content = document.getElementById('run-detail-content');
    content.innerHTML = `
        <div class="detail-section">
            <h3>Run Information</h3>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-item-label">Run ID</div>
                    <div class="detail-item-value">${escapeHtml(run.run_id || '-')}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Worker ID</div>
                    <div class="detail-item-value" style="display: flex; align-items: center; gap: 0.5rem;">
                        <span>${escapeHtml(run.worker_id || '-')}</span>
                        <button class="btn" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;" onclick="showWorkerDetail('${escapeHtml(run.worker_id)}')">View</button>
                    </div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Worker Role</div>
                    <div class="detail-item-value">${escapeHtml(run.expert_name || run.worker_role || '-')}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Status</div>
                    <div class="detail-item-value">
                        <span class="badge ${run.status === 'finished' ? 'badge-success' : run.status === 'crashed' ? 'badge-danger' : run.status === 'running' ? 'badge-info' : 'badge-warning'}">
                            ${escapeHtml(run.status || '-')}
                        </span>
                    </div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Unified Metric</div>
                    <div class="detail-item-value">${run.unified_metric ? run.unified_metric.toFixed(4) : '-'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Baseline Metric</div>
                    <div class="detail-item-value">${run.baseline_metric ? run.baseline_metric.toFixed(4) : '-'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Job ID</div>
                    <div class="detail-item-value">${escapeHtml(run.job_id || '-')}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Runtime</div>
                    <div class="detail-item-value">${formatRuntime(run.runtime_seconds)}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">GPU Util</div>
                    <div class="detail-item-value">${run.avg_gpu_util ? run.avg_gpu_util.toFixed(2) + '%' : '-'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">GPU Memory</div>
                    <div class="detail-item-value">${run.avg_gpu_memory ? run.avg_gpu_memory.toFixed(2) + ' GB' : '-'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Metric Delta</div>
                    <div class="detail-item-value">${run.metric_delta ? (run.metric_delta > 0 ? '+' : '') + run.metric_delta.toFixed(4) : '-'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Kept</div>
                    <div class="detail-item-value">${run.kept ? 'Yes' : 'No'}</div>
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <h3>Idea</h3>
            <div class="detail-item" style="margin-bottom: 1rem;">
                <div class="detail-item-label">Title</div>
                <div class="detail-item-value">${escapeHtml(run.idea_title || '-')}</div>
            </div>
            <div class="detail-item">
                <div class="detail-item-label">Outline</div>
                <div class="detail-item-value" style="white-space: pre-wrap;">${escapeHtml(run.idea_outline || '-')}</div>
            </div>
        </div>
        
        <div class="detail-section">
            <h3>Git Information</h3>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-item-label">Feature Branch</div>
                    <div class="detail-item-value"><code>${escapeHtml(run.feature_branch || '-')}</code></div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Baseline Commit</div>
                    <div class="detail-item-value"><code>${escapeHtml(run.baseline_commit || '-')}</code></div>
                </div>
                <div class="detail-item">
                    <div class="detail-item-label">Trial Commit</div>
                    <div class="detail-item-value"><code>${escapeHtml(run.trial_commit || '-')}</code></div>
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <h3>Summary</h3>
            <div class="summary-text">${escapeHtml(run.summary || 'No summary available')}</div>
        </div>
        
        ${parentHistory.length > 0 ? `
        <div class="detail-section">
            <h3>Parent History</h3>
            <div class="parent-history">
                ${parentHistory.map((parent, idx) => `
                    <div class="parent-entry">
                        <div class="parent-arrow">${idx > 0 ? '&#8593;' : ''}</div>
                        <div class="parent-content">
                            <div class="parent-header">
                                <span class="parent-index">${parentHistory.length - idx}</span>
                                <code class="parent-run-id">${escapeHtml(parent.run_id)}</code>
                                <span class="badge ${parent.status === 'finished' ? 'badge-success' : parent.status === 'crashed' ? 'badge-danger' : parent.status === 'running' ? 'badge-info' : 'badge-warning'}">${escapeHtml(parent.status || '-')}</span>
                                <button class="btn" style="margin-left: auto;" onclick="showRunDetail('${escapeHtml(parent.run_id)}')">View</button>
                            </div>
                            <div class="parent-idea">${escapeHtml(parent.idea_title || '-')}</div>
                            <div class="parent-meta">
                                <span>Worker: ${escapeHtml(parent.worker_id || '-')}</span>
                                <span>Metric: ${parent.unified_metric ? parent.unified_metric.toFixed(4) : '-'}</span>
                                <span>${formatTimestamp(parent.timestamp_utc)}</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
        ` : ''}
    `;
}

function initModal() {
    const modal = document.getElementById('modal');
    const closeBtn = document.getElementById('modal-close');
    
    closeBtn?.addEventListener('click', () => modal.classList.remove('active'));
    modal?.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });
}

function formatTimestamp(ts) {
    if (!ts) return '-';
    try {
        const date = new Date(ts);
        return date.toLocaleString();
    } catch {
        return ts;
    }
}

function formatRuntime(seconds) {
    if (seconds === null || seconds === undefined) return '-';
    const totalSeconds = Math.max(0, Math.floor(seconds));
    const days = Math.floor(totalSeconds / 86400);
    const hours = Math.floor((totalSeconds % 86400) / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;
    
    if (days > 0) {
        return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
}

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}