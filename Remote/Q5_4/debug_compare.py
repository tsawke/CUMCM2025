#!/usr/bin/env python3
import os,sys,math
import pandas as pd
sys.path.append('/root/q5')
import Q5ExSolver as ex

xlsx='/root/q5/smoke_optimization_result2.xlsx'
print('[INFO] reading', xlsx)
try:
    df = pd.read_excel(xlsx)
    print('[INFO] loaded', len(df), 'rows')
    print('[INFO] columns:', list(df.columns))
except Exception as e:
    print('[ERR] read_excel failed:', e)
    sys.exit(1)

# Try different column name patterns
possible_cols = {
    'drone': ['无人机编号', '烟幕弹编号', 'UAV', 'drone'],
    'speed': ['速度(m/s)', 'speed', 'Speed'],
    'direction': ['方向(°)', 'direction', 'Direction'],
    'drop_time': ['投放时刻(s)', 'drop_time', 'Drop Time'],
    'delay': ['起爆延迟(s)', 'delay', 'Delay'],
    'missile': ['干扰导弹', 'missile', 'Missile'],
    'effective': ['有效遮蔽时长(s)', 'effective_time', 'Effective Time']
}

# Find actual column names
actual_cols = {}
for key, candidates in possible_cols.items():
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    actual_cols[key] = found
    print(f'[INFO] {key} -> {found}')

# Check if we have minimum required columns
required = ['drone', 'speed', 'direction', 'drop_time', 'delay', 'missile']
missing = [k for k in required if actual_cols[k] is None]
if missing:
    print('[ERR] missing required columns:', missing)
    print('[INFO] Available columns:', list(df.columns))
    # Show first few rows to help debug
    print('[INFO] First 3 rows:')
    print(df.head(3))
    sys.exit(2)

# Extract data
rows = []
for i, r in df.iterrows():
    try:
        drone = str(r[actual_cols['drone']]).strip()
        speed = float(r[actual_cols['speed']])
        direction = float(r[actual_cols['direction']])
        drop_time = float(r[actual_cols['drop_time']])
        delay = float(r[actual_cols['delay']])
        missile = str(r[actual_cols['missile']]).strip()
        effective = float(r[actual_cols['effective']]) if actual_cols['effective'] and not pd.isna(r[actual_cols['effective']]) else float('nan')
        rows.append((drone, speed, direction, drop_time, delay, missile, effective))
    except Exception as e:
        print(f'[WARN] bad row {i}: {e}')

print(f'[INFO] extracted {len(rows)} valid rows')

# Evaluation function
def eval_mode(use_numba):
    ex.NUMBA_OK = use_numba
    # Reset drones
    for d in ex.DRONES:
        ex.DRONES[d]['smokes'] = []
        ex.DRONES[d]['speed'] = None
        ex.DRONES[d]['direction'] = None
    
    total = 0.0
    results = []
    
    # Sort by drone and drop time
    rows_sorted = sorted(rows, key=lambda x: (x[0], x[3]))
    
    for drone, speed, direction_deg, drop_time, delay, missile, ref_effective in rows_sorted:
        direction_rad = direction_deg * math.pi / 180.0
        ex.DRONES[drone]['speed'] = speed
        ex.DRONES[drone]['direction'] = direction_rad
        
        val = ex.calc_smoke_effective_time(drone, missile, drop_time, delay)
        ex.DRONES[drone]['smokes'].append({'drop_time': drop_time})
        
        total += val
        results.append((drone, missile, drop_time, delay, val, ref_effective))
    
    return total, results

# Compare both modes
print('\n[EVAL] Running numba mode...')
sum_numba, rows_numba = eval_mode(True)

print('[EVAL] Running python mode...')
sum_py, rows_py = eval_mode(False)

print(f'\n[SUMMARY]')
print(f'Numba total:  {sum_numba:.3f}s')
print(f'Python total: {sum_py:.3f}s')
print(f'Difference:   {sum_numba - sum_py:.3f}s')

print(f'\n[DETAILS]')
print('Drone Missile Drop  Delay  Numba  Python  Original')
for (a, b) in zip(rows_numba, rows_py):
    drone, missile, drop, delay, val_n, ref = a
    _, _, _, _, val_p, _ = b
    ref_str = f'{ref:.3f}' if not math.isnan(ref) else 'N/A'
    print(f'{drone:4s} {missile:2s}    {drop:5.2f} {delay:5.2f}  {val_n:5.3f}  {val_p:5.3f}   {ref_str}')

# Check if any individual calculations differ significantly
max_diff = 0
for (a, b) in zip(rows_numba, rows_py):
    diff = abs(a[4] - b[4])
    if diff > max_diff:
        max_diff = diff

print(f'\n[ANALYSIS]')
print(f'Max individual difference: {max_diff:.6f}s')
if max_diff > 1e-6:
    print('Significant numerical differences detected!')
else:
    print('Individual calculations are consistent.')

