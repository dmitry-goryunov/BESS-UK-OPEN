#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
import os
os.environ['PHASE4_DA_FORWARD_FEATURE_MODE'] = 'efa_blocks'

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
PROCESSED = PROJECT_ROOT / 'data' / 'processed'
EXECUTED_DIR = PROCESSED / 'executed'
EXECUTED_DIR.mkdir(parents=True, exist_ok=True)

NB12_PATH = NOTEBOOKS_DIR / '12_phase4_method_comparison.ipynb'

# ── Sweep configuration ──────────────────────────────────────────────────────
SWEEP_DURATIONS_H = [1.0, 2.0]
TIMEOUT_S = 10_800   # nbconvert safety cap; child wall-time cap below controls diagnostics
CHILD_WALLTIME_LIMIT_S = float(os.environ.get('PHASE4_CHILD_MAX_MINUTES', '45')) * 60
STATUS_POLL_S = float(os.environ.get('PHASE4_STATUS_POLL_S', '5'))
PROGRESS_INTERVAL_PCT = max(1.0, float(os.environ.get('PHASE4_PROGRESS_INTERVAL_PCT', '25')))

# Debug-sized child runs.  Set PHASE4_DEBUG=0 for the full 500-path medium run.
DEBUG_SWEEP = os.environ.get('PHASE4_DEBUG', '1').strip().lower() not in {'0', 'false', 'no'}
DEBUG_N_PATHS = int(os.environ.get('PHASE4_DEBUG_N_PATHS', '250'))
DEBUG_BWD_STEPS = int(os.environ.get('PHASE4_DEBUG_BWD_STEPS', '2160'))  # 45 days at HH resolution
DEBUG_FWD_WORKERS = int(os.environ.get('PHASE4_DEBUG_FWD_WORKERS', '1'))
FORCE_RERUN = os.environ.get('PHASE4_FORCE_RERUN', '1' if DEBUG_SWEEP else '0').strip().lower() not in {'0', 'false', 'no'}

# Default to 2 workers: faster than serial, less prone to child-kernel wedges
# and CPU oversubscription than launching all durations at once on Windows.
_default_parallel = str(min(1, len(SWEEP_DURATIONS_H)))
MAX_PARALLEL_DURATIONS = int(os.environ.get('PHASE4_MAX_PARALLEL', _default_parallel))
MAX_PARALLEL_DURATIONS = 1
_RI_WORKERS_EACH_DEFAULT = max(1, min(2, ((os.cpu_count() or 2) - 2) // max(2 * MAX_PARALLEL_DURATIONS, 1)))
RI_WORKERS_EACH = int(os.environ.get('PHASE4_RI_WORKERS_EACH', _RI_WORKERS_EACH_DEFAULT))
RI_WORKERS_EACH = max(1, RI_WORKERS_EACH)

print(f'Project root : {PROJECT_ROOT}')
print(f'Notebook 12  : {NB12_PATH}  (exists={NB12_PATH.exists()})')
print(f'Executed dir : {EXECUTED_DIR}')
print(f'Durations    : {SWEEP_DURATIONS_H}')
print(f'Timeout/run  : {TIMEOUT_S/3600:.1f} h nbconvert cap; {CHILD_WALLTIME_LIMIT_S/60:.1f} min child wall cap')
print(f'Progress step: {PROGRESS_INTERVAL_PCT:g}% status buckets')
print(f'Max parallel : {MAX_PARALLEL_DURATIONS}')
print(f'RI workers   : {RI_WORKERS_EACH} each for DA/WD per child')
print(f'Force rerun  : {FORCE_RERUN}')
if DEBUG_SWEEP:
    print(f'Debug sweep  : ON  paths={DEBUG_N_PATHS}, bwd_steps={DEBUG_BWD_STEPS}, fwd_workers={DEBUG_FWD_WORKERS}')
else:
    print('Debug sweep  : OFF  full medium settings')


# ## 2  Sequential Execution
# 

# In[3]:


from concurrent.futures import ThreadPoolExecutor, as_completed

STALE_DEPENDENCIES = [
    NB12_PATH,
    PROJECT_ROOT / 'src' / 'optimisation' / 'lsmc.py',
    PROJECT_ROOT / 'src' / 'optimisation' / 'dispatch.py',
    PROJECT_ROOT / 'src' / 'optimisation' / 'rolling_intrinsic.py',
    PROCESSED / 'sim_bundle.pkl',
    PROCESSED / 'imbalance_params.json',
    PROCESSED / 'ss_params.json',
    PROCESSED / 'pca_params.json',
    PROCESSED / 'ancillary_params.json',
]

def _latest_dependency_mtime() -> float:
    existing = [p.stat().st_mtime for p in STALE_DEPENDENCIES if p.exists()]
    return max(existing) if existing else 0.0

def _tail_text(path: Path, max_chars: int = 3000) -> str:
    if not path.exists():
        return ''
    text = path.read_text(encoding='utf-8', errors='replace')
    return text[-max_chars:]

def _read_status_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events

def _terminate_process_tree(proc: subprocess.Popen, label: str) -> None:
    if proc.poll() is not None:
        return
    print(f'[{label}] terminating child process tree (pid={proc.pid})')
    if os.name == 'nt':
        subprocess.run(
            ['taskkill', '/PID', str(proc.pid), '/T', '/F'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.terminate()

def _run_one_duration(dur: float) -> tuple[str, dict]:
    label = f'{dur:g}h'
    out_nb = EXECUTED_DIR / f'12_phase4_{label}.ipynb'
    csv_out = PROCESSED / f'phase4_method_comparison_{label}.csv'
    status_path = PROCESSED / f'phase4_status_{label}.jsonl'
    log_path = PROCESSED / f'phase4_child_{label}.log'

    if csv_out.exists() and out_nb.exists() and not FORCE_RERUN:
        output_mtime = min(csv_out.stat().st_mtime, out_nb.stat().st_mtime)
        dependency_mtime = _latest_dependency_mtime()
        if output_mtime >= dependency_mtime:
            print(f'[{label}] outputs exist and are current ? skipping')
            return label, {'status': 'skipped', 'elapsed_s': 0}
        print(f'[{label}] outputs are stale after code/notebook changes ? re-running')
    elif csv_out.exists() and out_nb.exists() and FORCE_RERUN:
        print(f'[{label}] outputs exist but FORCE_RERUN=True ? re-running')

    for path in (status_path, log_path):
        if path.exists():
            path.unlink()

    print(f'[{label}] Starting  {pd.Timestamp.now().strftime("%H:%M:%S")}')
    t0 = time.time()

    env = os.environ.copy()
    env['NB12_DURATION_H'] = str(dur)
    env['NB12_WD_UPLIFT_CAP_GBP_MWH'] = os.environ.get('PHASE4_WD_UPLIFT_CAP_GBP_MWH', '30')
    env['NB12_STATUS_PATH'] = str(status_path)
    env['NB12_PROGRESS_INTERVAL_PCT'] = f'{PROGRESS_INTERVAL_PCT:g}'
    env['NB12_DA_FORWARD_FEATURE_HH'] = os.environ.get('PHASE4_DA_FORWARD_FEATURE_HH', '48')
    env['NB12_DA_FORWARD_FEATURE_MODE'] = os.environ.get('PHASE4_DA_FORWARD_FEATURE_MODE', 'summary')
    env['NB12_IMBALANCE_SIGNAL_LAG_HH'] = os.environ.get('PHASE4_IMBALANCE_SIGNAL_LAG_HH', '1')
    env['NB12_ALLOW_ANCILLARY_STACKING'] = os.environ.get('PHASE4_ALLOW_ANCILLARY_STACKING', '0')
    env['NB12_RESERVE_SUSTAIN_H'] = os.environ.get('PHASE4_RESERVE_SUSTAIN_H', '1.0')
    env['NB12_RI_WORKERS_EACH'] = str(RI_WORKERS_EACH)
    if DEBUG_SWEEP:
        env['NB12_RI_PATHS'] = str(DEBUG_N_PATHS)
        env['NB12_BWD_PATHS'] = str(DEBUG_N_PATHS)
        env['NB12_FWD_PATHS'] = str(DEBUG_N_PATHS)
        env['NB12_BWD_STEPS'] = str(DEBUG_BWD_STEPS)
        env['NB12_FWD_WORKERS'] = str(DEBUG_FWD_WORKERS)

    cmd = [
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        f'--ExecutePreprocessor.timeout={TIMEOUT_S}',
        f'--output={out_nb}',
        str(NB12_PATH),
    ]

    last_bucket_by_phase: dict[str, float] = {}
    last_event_s = 0.0
    last_heartbeat_s = 0.0
    timed_out = False
    with log_path.open('w', encoding='utf-8') as log_fh:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while proc.poll() is None:
            elapsed = time.time() - t0
            printed_event = False
            events = _read_status_events(status_path)
            for event in events:
                phase = str(event.get('phase', 'unknown'))
                pct = float(event.get('progress_pct', 0.0))
                bucket = 100.0 if pct >= 100.0 else int(pct // PROGRESS_INTERVAL_PCT) * PROGRESS_INTERVAL_PCT
                if bucket > last_bucket_by_phase.get(phase, -1):
                    last_bucket_by_phase[phase] = bucket
                    step = int(event.get('step', 0))
                    total = int(event.get('total_steps', 0))
                    ev_elapsed = float(event.get('elapsed_s', elapsed))
                    print(
                        f'[{label}] {phase:8s} {bucket:3.0f}%  '
                        f'({step:,}/{total:,} steps, {ev_elapsed/60:.1f} min)'
                    )
                    printed_event = True
            if printed_event:
                last_event_s = elapsed
            elif elapsed - last_heartbeat_s >= 60.0:
                last_heartbeat_s = elapsed
                if events:
                    latest = events[-1]
                    latest_phase = str(latest.get('phase', 'unknown'))
                    latest_pct = float(latest.get('progress_pct', 0.0))
                    latest_age = elapsed - float(latest.get('elapsed_s', elapsed))
                    print(
                        f'[{label}] still running {elapsed/60:.1f} min; '
                        f'last status {latest_phase} {latest_pct:.0f}% '
                        f'({latest_age/60:.1f} min ago)'
                    )
                else:
                    print(f'[{label}] still running {elapsed/60:.1f} min; no status yet')
            if elapsed >= CHILD_WALLTIME_LIMIT_S:
                timed_out = True
                print(f'[{label}] TIME LIMIT {CHILD_WALLTIME_LIMIT_S/60:.1f} min reached')
                _terminate_process_tree(proc, label)
                break
            time.sleep(STATUS_POLL_S)

        try:
            returncode = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(proc, label)
            returncode = proc.wait(timeout=30)

    elapsed = time.time() - t0
    status = 'timeout' if timed_out else ('ok' if returncode == 0 else 'failed')

    if returncode == 0 and not timed_out:
        print(f'[{label}] DONE in {elapsed/60:.1f} min  ?  {out_nb.name}')
    else:
        print(f'[{label}] {status.upper()} (rc={returncode}) after {elapsed/60:.1f} min')
        print(f'--- child log tail ({log_path.name}) ---')
        print(_tail_text(log_path, 3000))

    return label, {
        'status': status,
        'elapsed_s': round(elapsed, 1),
        'output_nb': str(out_nb),
        'status_path': str(status_path),
        'log_path': str(log_path),
        'returncode': returncode,
    }


# Run all durations in parallel ? each is a separate subprocess on its own core.
# Wall-clock time ? slowest single duration instead of sum of all durations.
print(f'Launching {len(SWEEP_DURATIONS_H)} durations in parallel...\n')
t_sweep_start = time.time()

run_log = {}
with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DURATIONS) as pool:
    futures = {pool.submit(_run_one_duration, dur): dur for dur in SWEEP_DURATIONS_H}
    for fut in as_completed(futures):
        label, result = fut.result()
        run_log[label] = result

print(f'\n\nRun summary  (total wall-clock: {(time.time()-t_sweep_start)/60:.1f} min):')
for label in [f'{d:g}h' for d in SWEEP_DURATIONS_H]:
    r = run_log.get(label, {})
    mins = r.get('elapsed_s', 0) / 60
    print(f'  {label:5s}  {r.get("status","?"):8s}  {mins:.1f} min')

# Save run log
with open(PROCESSED / 'phase4_sweep_run_log.json', 'w') as f:
    json.dump(run_log, f, indent=2)
print('Saved: phase4_sweep_run_log.json')


# ## 3  Load and Aggregate Results
# 

# In[4]:


all_comparison  = []
all_attribution = []
missing = []

for dur in SWEEP_DURATIONS_H:
    label = f'{dur:g}h'

    csv_path = PROCESSED / f'phase4_method_comparison_{label}.csv'
    if csv_path.exists():
        all_comparison.append(pd.read_csv(csv_path))
        print(f'  loaded {csv_path.name}')
    else:
        missing.append(str(csv_path.name))
        print(f'  MISSING {csv_path.name}')

    attr_path = PROCESSED / f'lsmc_attribution_{label}.json'
    if attr_path.exists():
        with open(attr_path) as fh:
            rows = json.load(fh)
        # ensure duration_h is present (nb12 now writes it; older files may not)
        for r in rows:
            r.setdefault('duration_h', dur)
        all_attribution.extend(rows)
        print(f'  loaded {attr_path.name}')
    else:
        missing.append(str(attr_path.name))
        print(f'  MISSING {attr_path.name}')

if missing:
    print(f'\nWARNING: {len(missing)} files missing — re-run section 2 to generate them')

df_cmp  = pd.concat(all_comparison,  ignore_index=True) if all_comparison  else pd.DataFrame()
df_attr = pd.DataFrame(all_attribution) if all_attribution else pd.DataFrame()

print(f'\nComparison table : {len(df_cmp)} rows,  {df_cmp["duration_h"].nunique() if len(df_cmp) else 0} durations')
print(f'Attribution table: {len(df_attr)} rows')
df_cmp.head(10)


# ## 4  Method Comparison Across Durations
# 

# In[5]:


METHODS = [
    'Initial hourly intrinsic',
    'DA rolling intrinsic',
    'WD rolling intrinsic',
    'Forward simulation (LSMC)',
    'Perfect foresight (DA energy)',
]
METHOD_COLOURS = {
    'Initial hourly intrinsic':      '#aec7e8',
    'DA rolling intrinsic':          '#2c5f8a',
    'WD rolling intrinsic':          '#1a3a57',
    'Forward simulation (LSMC)':     '#f28e2b',
    'Perfect foresight (DA energy)': '#4e79a7',
}
METHOD_MARKERS = {
    'Initial hourly intrinsic':      's',
    'DA rolling intrinsic':          'o',
    'WD rolling intrinsic':          'D',
    'Forward simulation (LSMC)':     '*',
    'Perfect foresight (DA energy)': '^',
}

durations_avail = sorted(df_cmp['duration_h'].unique()) if len(df_cmp) else []

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Phase 4 method comparison — duration sweep  (medium run mode)', fontsize=12)

# ── Left: grouped bar by duration ────────────────────────────────────────────
ax = axes[0]
n_dur = len(durations_avail)
n_met = len(METHODS)
x     = np.arange(n_dur)
width = 0.8 / n_met

for i, method in enumerate(METHODS):
    sub  = df_cmp[df_cmp['method'] == method].set_index('duration_h')
    vals = [float(sub.loc[d, 'value_gbp_annualized_m']) if d in sub.index else np.nan for d in durations_avail]
    offsets = x + i * width - (n_met - 1) * width / 2
    ax.bar(offsets, vals, width * 0.92,
           color=METHOD_COLOURS.get(method, '#888888'), alpha=0.85, label=method)

ax.set_xticks(x)
ax.set_xticklabels([f'{d:g}h' for d in durations_avail])
ax.set_xlabel('Battery duration')
ax.set_ylabel('GBP million / year (annualised)')
ax.set_title('Value by method and duration', fontsize=10)
ax.legend(fontsize=7, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# ── Right: line chart — value vs duration ────────────────────────────────────
ax = axes[1]
for method in METHODS:
    sub = df_cmp[df_cmp['method'] == method].sort_values('duration_h')
    if sub.empty:
        continue
    col = METHOD_COLOURS.get(method, '#888888')
    mrk = METHOD_MARKERS.get(method, 'o')
    ax.plot(sub['duration_h'], sub['value_gbp_annualized_m'],
            marker=mrk, linewidth=1.8, markersize=6, color=col, label=method)
    if 'p5_ann_m' in sub.columns and 'p95_ann_m' in sub.columns:
        ax.fill_between(sub['duration_h'],
                        sub['p5_ann_m'], sub['p95_ann_m'],
                        alpha=0.10, color=col)

ax.set_xlabel('Battery duration (hours)')
ax.set_ylabel('GBP million / year (annualised)')
ax.set_title('Value vs duration  [bands: ±1 std]', fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
if durations_avail:
    ax.set_xticks(durations_avail)

fig.tight_layout()
out_cmp_png = PROCESSED / 'phase4_all_durations_comparison.png'
fig.savefig(out_cmp_png, dpi=140, bbox_inches='tight')
print(f'Saved: {out_cmp_png}')
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
if len(df_cmp):
    pivot = df_cmp.pivot_table(
        index='method', columns='duration_h',
        values='value_gbp_annualized_m', aggfunc='first',
    ).reindex(METHODS)
    pivot.columns = [f'{c:g}h' for c in pivot.columns]
    print('\nLSMC value (£m/year) by method and duration:')
    print(pivot.round(2).to_string())


# ## 5  LSMC Attribution Across Durations
# 

# In[6]:


COMPONENTS_ORD = ['HPFC anchor', 'DA surprise', 'Imbalance proxy (BM/ID substitute)', 'DC ancillary', 'QR ancillary', 'Balancing Mechanism', 'Costs (deg+VOM)']
COMP_COLOURS = {
    'HPFC anchor':      '#aec7e8',
    'DA surprise':      '#4e79a7',
    'Imbalance proxy (BM/ID substitute)': '#76b7b2',
    'DC ancillary':     '#59a14f',
    'QR ancillary':     '#edc948',
    'Balancing Mechanism': '#f28e2b',
    'Costs (deg+VOM)':  '#e15759',
}

if df_attr.empty:
    print('No attribution data loaded — skipping chart')
else:
    dur_list = sorted(df_attr['duration_h'].unique())
    x        = np.arange(len(dur_list))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('LSMC value attribution — duration sweep  (medium run mode)', fontsize=12)

    # ── Left: grouped bar — component mean by duration ────────────────────────
    ax = axes[0]
    n_comp = len(COMPONENTS_ORD)
    width  = 0.8 / n_comp

    for i, comp in enumerate(COMPONENTS_ORD):
        sub  = df_attr[df_attr['component'] == comp].set_index('duration_h')
        vals = [float(sub.loc[d, 'mean_m']) if d in sub.index else np.nan for d in dur_list]
        offsets = x + i * width - (n_comp - 1) * width / 2
        ax.bar(offsets, vals, width * 0.92,
               color=COMP_COLOURS.get(comp, '#888'), alpha=0.85, label=comp)

    ax.axhline(0, color='#555', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:g}h' for d in dur_list])
    ax.set_xlabel('Battery duration')
    ax.set_ylabel('GBP million / year (annualised)')
    ax.set_title('Attribution by component and duration', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # ── Right: stacked bar — gross revenue mix by duration ────────────────────
    ax    = axes[1]
    gross = [c for c in COMPONENTS_ORD if c != 'Costs (deg+VOM)']
    bot   = np.zeros(len(dur_list))

    for comp in gross:
        sub  = df_attr[df_attr['component'] == comp].set_index('duration_h')
        vals = np.array([
            max(float(sub.loc[d, 'mean_m']), 0.0) if d in sub.index else 0.0
            for d in dur_list
        ])
        ax.bar(x, vals, bottom=bot,
               color=COMP_COLOURS[comp], alpha=0.85, label=comp)
        bot += vals

    # Overlay total LSMC line
    if len(df_cmp):
        lsmc_sub = df_cmp[df_cmp['method'] == 'Forward simulation (LSMC)'].set_index('duration_h')
        totals   = [float(lsmc_sub.loc[d, 'value_gbp_annualized_m']) if d in lsmc_sub.index else np.nan
                    for d in dur_list]
        ax.plot(x, totals, 'o--', color='#333333', linewidth=1.5, markersize=5, label='LSMC net total')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:g}h' for d in dur_list])
    ax.set_xlabel('Battery duration')
    ax.set_ylabel('GBP million / year (annualised)')
    ax.set_title('Gross revenue mix by duration (stacked)', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out_attr_png = PROCESSED / 'phase4_all_durations_attribution.png'
    fig.savefig(out_attr_png, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_attr_png}')
    plt.show()

    # Attribution share table
    print('\nAttribution share (% of gross revenue):')
    attr_pivot = df_attr[df_attr['component'].isin(gross)].pivot_table(
        index='component', columns='duration_h',
        values='pct_of_gross', aggfunc='first',
    ).reindex([c for c in COMPONENTS_ORD if c in gross])
    attr_pivot.columns = [f'{c:g}h' for c in attr_pivot.columns]
    print(attr_pivot.round(1).to_string())


# ## 6  Save Combined Outputs
# 

# In[7]:


saved_outputs = []

def _save_csv_json(frame: pd.DataFrame, stem: str) -> None:
    out_csv = PROCESSED / f'{stem}.csv'
    frame.to_csv(out_csv, index=False)
    saved_outputs.append(out_csv)
    print(f'Saved: {out_csv}')

    out_json = PROCESSED / f'{stem}.json'
    frame.to_json(out_json, orient='records', indent=2)
    saved_outputs.append(out_json)
    print(f'Saved: {out_json}')

if len(df_cmp):
    _save_csv_json(df_cmp, 'phase4_all_durations_comparison')

if len(df_attr):
    _save_csv_json(df_attr, 'phase4_all_durations_attribution')

missing = [p for p in saved_outputs if not p.exists() or p.stat().st_size == 0]
if missing:
    raise FileNotFoundError('Save verification failed: ' + ', '.join(str(p) for p in missing))

print(f'\nAll outputs saved and verified ({len(saved_outputs)} files).')

