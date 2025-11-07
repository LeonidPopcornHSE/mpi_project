import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('default')

df = pd.read_csv('results.csv')

os.makedirs('plots', exist_ok=True)

points_list = sorted(df['points'].unique())
procs_list = sorted(df['procs'].unique())

print("Загружены данные:")
print(f"Количество процессов: {procs_list}")
print(f"Размеры задач: {points_list}")

print("Вычисляем ускорение...")
speedups = []

for points in points_list:
    subset = df[df['points'] == points]
    time_1proc = subset[subset['procs'] == 1]['time'].values[0]
    
    for _, row in subset.iterrows():
        if row['procs'] == 1:
            speedup = 1.0
        else:
            speedup = time_1proc / row['time']
        speedups.append(speedup)

df['speedup'] = speedups

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for points in points_list:
    subset = df[df['points'] == points]
    ax1.plot(subset['procs'], subset['time'], 
             marker='o', linewidth=2, markersize=6,
             label=f'{points:,} точек')

ax1.set_xlabel('Количество процессов', fontsize=12)
ax1.set_ylabel('Время выполнения (секунды)', fontsize=12)
ax1.set_title('Время выполнения', fontsize=14)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(procs_list)

for points in points_list:
    subset = df[df['points'] == points]
    ax2.plot(subset['procs'], subset['speedup'],
             marker='s', linewidth=2, markersize=6,
             label=f'{points:,} точек')

ax2.plot(procs_list, procs_list, 'k--', label='Идеальное ускорение', alpha=0.7, linewidth=2)

ax2.set_xlabel('Количество процессов', fontsize=12)
ax2.set_ylabel('Ускорение', fontsize=12)
ax2.set_title('Ускорение', fontsize=14)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(procs_list)

plt.tight_layout()
plt.savefig('plots/summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*90)
print("ТАБЛИЦА РЕЗУЛЬТАТОВ МЕТОДА МОНТЕ-КАРЛО")
print("="*90)
print("Формат: время (ускорение)")
print("="*90)

header = "Количество точек".ljust(20)
for procs in procs_list:
    if procs == 1:
        header += " | 1 процесс".ljust(15)
    else:
        header += f" | {procs} процессов".ljust(15)
print(header)
print("-" * len(header))

for points in points_list:
    subset = df[df['points'] == points].sort_values('procs')
    
    row = f"{points:,}".ljust(20)
    
    for procs in procs_list:
        data = subset[subset['procs'] == procs]
        if not data.empty:
            time_val = data['time'].values[0]
            speedup_val = data['speedup'].values[0]
            
            if procs == 1:
                cell = f"{time_val:.4f}".ljust(15)
            else:
                cell = f"{time_val:.4f} ({speedup_val:.2f})".ljust(15)
        else:
            cell = " ".ljust(15)
        
        row += " | " + cell
    
    print(row)
    print("-" * len(header))

print("\n" + "="*50)
print("КРАТКИЙ АНАЛИЗ")
print("="*50)

print("Анализ ускорения:")
for points in points_list:
    subset = df[df['points'] == points]
    max_procs = max(subset['procs'])
    if max_procs > 1:
        speedup_max = subset[subset['procs'] == max_procs]['speedup'].values[0]
        ideal = max_procs
        efficiency = (speedup_max / ideal) * 100
        print(f"• {points:,} точек: ускорение на {max_procs} процессах = {speedup_max:.2f} (эффективность: {efficiency:.1f}%)")

print(f"\nГрафик сохранен: 'plots/summary.png'")
print("Таблица результатов выведена выше")
