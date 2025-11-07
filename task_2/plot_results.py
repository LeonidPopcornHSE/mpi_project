# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
 
# read
df = pd.read_csv("results.csv")
 
# expected columns: mode,N,procs,avg_time
required = {'mode','N','procs','avg_time'}
if not required.issubset(df.columns):
    raise SystemExit(f"results.csv must contain columns: {required}. Found: {list(df.columns)}")
 
modes = sorted(df['mode'].unique())
sizes = sorted(df['N'].unique())
 
# compute speedup and efficiency using p==1 per (mode,N) as baseline
rows = []
for mode in modes:
    for N in sizes:
        sub = df[(df['mode']==mode) & (df['N']==N)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('procs')
        t1_vals = sub[sub['procs']==1]['avg_time'].values
        if len(t1_vals)==0:
            # no p=1 baseline for this mode,N -> skip speedup/eff
            sub['speedup'] = np.nan
            sub['efficiency'] = np.nan
        else:
            t1 = float(t1_vals[0])
            sub['speedup'] = t1 / sub['avg_time'].values
            sub['efficiency'] = sub['speedup'] / sub['procs'].values
        rows.append(sub)
 
if len(rows)==0:
    raise SystemExit("No data found after grouping.")
 
df_full = pd.concat(rows, ignore_index=True)
df_full = df_full[['mode','N','procs','avg_time','speedup','efficiency']]
df_full.to_csv("results_with_speed.csv", index=False)
print("Saved results_with_speed.csv")
 
# plotting
os.makedirs("plots", exist_ok=True)
for N in sizes:
    sub_all = df_full[df_full['N']==N]
    if sub_all.empty:
        continue
 
    plt.figure(figsize=(6,4))
    for mode in modes:
        sub = sub_all[sub_all['mode']==mode].sort_values('procs')
        if sub.empty: continue
        plt.plot(sub['procs'], sub['avg_time'], marker='o', label=mode)
    # use log x axis if more than one distinct procs and all >0
    procs_vals = sorted(sub_all['procs'].unique())
    if len(procs_vals) > 1:
        try:
            plt.xscale('log', base=2)
        except TypeError:
            plt.xscale('log')
    plt.xlabel('Processes')
    plt.ylabel('Time (s)')
    plt.title(f'Execution time, N={N}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/time_N_{N}.png", bbox_inches='tight')
    plt.close()
 
    plt.figure(figsize=(6,4))
    for mode in modes:
        sub = sub_all[sub_all['mode']==mode].sort_values('procs')
        if sub.empty or sub['speedup'].isna().all(): continue
        plt.plot(sub['procs'], sub['speedup'], marker='o', label=f"{mode} speedup")
        plt.plot(sub['procs'], sub['efficiency'], marker='x', linestyle='--', label=f"{mode} efficiency")
    if len(procs_vals) > 1:
        try:
            plt.xscale('log', base=2)
        except TypeError:
            plt.xscale('log')
    plt.xlabel('Processes')
    plt.ylabel('Speedup / Efficiency')
    plt.title(f'Speedup & Efficiency, N={N}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/speed_eff_N_{N}.png", bbox_inches='tight')
    plt.close()
 
    # Build a table like the screenshot: rows = modes, columns grouped by procs
    procs_sorted = sorted(sub_all['procs'].unique())
    # Create a DataFrame: index = modes, for each proc two columns (time, speedup)
    cols = []
    for p in procs_sorted:
        cols.append((f"time_p{p}", f"time_p{p}"))
        cols.append((f"speed_p{p}", f"speed_p{p}"))
    # build table data
    table_df = pd.DataFrame(index=modes, columns=[c for pair in cols for c in pair])
    for mode in modes:
        for p in procs_sorted:
            row = sub_all[(sub_all['mode']==mode) & (sub_all['procs']==p)]
            if not row.empty:
                table_df.at[mode, f"time_p{p}"] = f"{float(row['avg_time']):.6g}"
                sp = row['speedup'].values[0]
                if np.isfinite(sp):
                    table_df.at[mode, f"speed_p{p}"] = f"{float(sp):.4f}"
                else:
                    table_df.at[mode, f"speed_p{p}"] = ""
            else:
                table_df.at[mode, f"time_p{p}"] = ""
                table_df.at[mode, f"speed_p{p}"] = ""
 
    # save numeric table CSV for this N
    table_df.to_csv(f"plots/table_N_{N}.csv")
    # render table as PNG
    fig, ax = plt.subplots(figsize=(max(6, 1.2*len(procs_sorted)*2), 1.5 + 0.6*len(modes)))
    ax.axis('off')
    tbl = ax.table(cellText=table_df.fillna("").values,
                   rowLabels=table_df.index,
                   colLabels=table_df.columns,
                   cellLoc='center',
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)
    ax.set_title(f"Results table N={N}", pad=12)
    plt.tight_layout()
    plt.savefig(f"plots/table_N_{N}.png", bbox_inches='tight')
    plt.close()
 
print("Plots and tables saved in 'plots/' directory.")
