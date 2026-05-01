"""
run_phase1.py — Phase 1 data pipeline (plain Python, no Jupyter required)

Run from the project root:
    cd "G:\My Drive\Research\bess_project"
    python notebooks/run_phase1.py

Or with custom dates:
    python notebooks/run_phase1.py --start 2024-04-01 --end 2026-04-25

Outputs saved to data/raw/
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Install dependencies if missing ──────────────────────────────────────────
import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install',
                'requests', 'pandas', 'pyarrow', 'matplotlib', '--quiet'])

from datetime import date
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--start', default='2024-04-01')
parser.add_argument('--end',   default='2026-04-25')
args = parser.parse_args()

START   = date.fromisoformat(args.start)
END     = date.fromisoformat(args.end)
RAW_DIR = Path(__file__).parent.parent / 'data' / 'raw'
PLT_DIR = Path(__file__).parent.parent / 'data' / 'processed' / 'plots'
RAW_DIR.mkdir(parents=True, exist_ok=True)
PLT_DIR.mkdir(parents=True, exist_ok=True)

print(f'\nBESS Phase 1 Data Pipeline')
print(f'Date range : {START} to {END}')
print(f'Output dir : {RAW_DIR}')
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Elexon day-ahead prices
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 55)
print('STEP 1/4  Elexon day-ahead prices (MID)')
print('=' * 55)

from src.data.fetch_elexon import fetch_mid_range

da_path = RAW_DIR / 'elexon_da_prices.parquet'
df_da   = fetch_mid_range(START, END, out_path=da_path)

if not df_da.empty:
    y2025 = df_da[df_da['settlement_date'].dt.year == 2025]['price_gbp_mwh']
    print(f'\n  Rows      : {len(df_da):,}')
    print(f'  2025 mean : £{y2025.mean():.1f}/MWh  (SS anchor: £76.7)')
    print(f'  2025 std  : £{y2025.std():.1f}/MWh')
    print(f'  Neg HHs   : {(df_da["price_gbp_mwh"] < 0).sum()}  (arithmetic OU required)')

    daily = df_da.groupby('settlement_date')['price_gbp_mwh'].mean()
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(daily.index, daily.values, lw=0.8, color='steelblue')
    ax.set(title='GB Day-Ahead Price — Daily Average (GBP/MWh)', ylabel='GBP/MWh')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(PLT_DIR / '01_da_prices.png', dpi=120)
    plt.close()
    print(f'  Plot      : data/processed/plots/01_da_prices.png')
else:
    print('  WARNING: No data returned — check network / Elexon API status')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Elexon System Price
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 55)
print('STEP 2/4  Elexon System Price (imbalance / cash-out)')
print('=' * 55)

from src.data.fetch_elexon import fetch_system_prices_range

sp_path = RAW_DIR / 'elexon_sp_prices.parquet'
df_sp   = fetch_system_prices_range(START, END, out_path=sp_path)

if not df_sp.empty and not df_da.empty:
    merged = df_da[['settlement_date','settlement_period','price_gbp_mwh']].merge(
        df_sp[['settlement_date','settlement_period','system_price','net_imbalance_volume']],
        on=['settlement_date','settlement_period'], how='inner'
    )
    merged['imbalance_basis'] = merged['system_price'] - merged['price_gbp_mwh']

    print(f'\n  Rows            : {len(df_sp):,}')
    print(f'  Imbalance basis : mean={merged["imbalance_basis"].mean():.1f}  '
          f'std={merged["imbalance_basis"].std():.1f}  '
          f'min={merged["imbalance_basis"].min():.1f}  '
          f'max={merged["imbalance_basis"].max():.1f}')
    print(f'  Jump HHs (|Δ|>100 GBP/MWh): {(merged["imbalance_basis"].abs() > 100).sum()}')

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(merged.groupby('settlement_date')['system_price'].mean(),
                 lw=0.8, color='darkorange')
    axes[0].set(title='System Price — Daily Average (GBP/MWh)', ylabel='GBP/MWh')
    axes[1].plot(merged.groupby('settlement_date')['imbalance_basis'].mean(),
                 lw=0.8, color='crimson')
    axes[1].axhline(0, color='k', lw=0.5, ls='--')
    axes[1].set(title='Imbalance Basis (SP - DA) — Daily Average', ylabel='GBP/MWh')
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(PLT_DIR / '02_imbalance_basis.png', dpi=120)
    plt.close()
    print(f'  Plot            : data/processed/plots/02_imbalance_basis.png')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — NESO EAC ancillary clearing
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 55)
print('STEP 3/4  NESO EAC ancillary clearing')
print('=' * 55)

from src.data.fetch_neso import fetch_all_ancillary

anc_path = RAW_DIR / 'neso_eac_clearing.parquet'
df_anc   = fetch_all_ancillary(START.isoformat(), END.isoformat(), out_path=anc_path)

if not df_anc.empty:
    print(f'\n  Rows: {len(df_anc):,}')
    print(df_anc.groupby('product')['clearing_price_gbp_mwh']
               .agg(['count','mean','min','max']).round(2).to_string())

    if 'DC' in df_anc['product'].values:
        dc_daily = (df_anc[df_anc['product']=='DC']
                    .groupby('date')['clearing_price_gbp_mwh'].mean())
        fig, ax  = plt.subplots(figsize=(12, 3))
        ax.plot(dc_daily.index, dc_daily.values, lw=0.8, color='teal')
        ax.axhline(17, color='gray', lw=0.8, ls='--', label='2020-21 cap £17/MW/h')
        ax.set(title='DC Clearing Price — Daily Average (GBP/MW/h)', ylabel='GBP/MW/h')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        fig.savefig(PLT_DIR / '03_dc_clearing.png', dpi=120)
        plt.close()
        print(f'\n  Plot: data/processed/plots/03_dc_clearing.png')
else:
    print('\n  WARNING: No ancillary data returned.')
    print('  Action : go to https://api.nationalgrideso.com')
    print('           search "Dynamic Containment EFA", copy the resource ID')
    print('           update RESOURCE_IDS in src/data/fetch_neso.py')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Forward curve (synthetic)
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 55)
print('STEP 4/4  Forward curve (synthetic — replace with ICE/EEX)')
print('=' * 55)

from src.data.fetch_forwards import build_synthetic_forwards

fwd_path = RAW_DIR / 'ice_eex_forwards.parquet'
df_fwd   = build_synthetic_forwards(as_of=END)
df_fwd.to_parquet(fwd_path, index=False)

print(f'\n  Rows: {len(df_fwd)}')
print(df_fwd[['contract','delivery_start','price_gbp_mwh']].to_string(index=False))
print()
print('  To replace with real forwards:')
print('  ICE: python -m src.data.fetch_forwards --source ice --file <export.csv>')
print('  EEX: python -m src.data.fetch_forwards --source eex --file <export.csv>')

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df_fwd['delivery_start'], df_fwd['price_gbp_mwh'],
        'o-', color='navy', ms=5, lw=1.2)
ax.axhline(76.7, color='gray', lw=0.8, ls='--', label='KYOS anchor £76.7/MWh')
ax.set(title='GB Baseload Forward Curve (synthetic)', ylabel='GBP/MWh')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
fig.savefig(PLT_DIR / '04_forward_curve.png', dpi=120)
plt.close()
print(f'  Plot: data/processed/plots/04_forward_curve.png')

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 55)
print('VALIDATION SUMMARY')
print('=' * 55)

checks = {
    'elexon_da_prices.parquet':  100,
    'elexon_sp_prices.parquet':  100,
    'neso_eac_clearing.parquet':   0,
    'ice_eex_forwards.parquet':    1,
}

all_ok = True
for name, min_rows in checks.items():
    path = RAW_DIR / name
    if path.exists():
        n   = len(pd.read_parquet(path))
        ok  = n >= min_rows
        sym = 'OK  ' if ok else 'WARN'
        print(f'  [{sym}]  {name}: {n:,} rows')
        if not ok: all_ok = False
    else:
        print(f'  [FAIL]  {name}: MISSING')
        all_ok = False

print()
print('RESULT:', 'All files present — ready for Phase 2 calibration'
      if all_ok else 'Some files missing — see warnings above')
print()
print('Next step: notebooks/02_calibration.ipynb  (or run_phase2.py when built)')
