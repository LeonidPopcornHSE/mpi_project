import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('default')

df = pd.read_csv('matrix_results.csv')

os.makedirs('matrix_plots', exist_ok=True)

matrix_sizes = sorted(df['matrix_size'].unique())
procs_list = sorted(df['procs'].unique())

print("Загружены данные:")
print(f"Количество процессов: {procs_list}")
print(f"Размеры матриц: {matrix_sizes}")

print("Вычисляем ускорение...")
speedups = []

for size in matrix_sizes:
    subset = df[df['matrix_size'] == size]
    time_1proc = subset[subset['procs'] == 1]['time'].values[0]
    
    for _, row in subset.iterrows():
        if row['procs'] == 1:
            speedup = 1.0
        else:
            speedup = time_1proc / row['time']
        speedups.append(speedup)

df['speedup'] = speedups

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for procs in procs_list:
    subset = df[df['procs'] == procs]
    ax1.plot(subset['matrix_size'], subset['time'], 
             marker='o', linewidth=2, markersize=6,
             label=f'{procs} процессов')

ax1.set_xlabel('Размер матрицы', fontsize=12)
ax1.set_ylabel('Время выполнения (секунды)', fontsize=12)
ax1.set_title('Время выполнения алгоритма Кэннона', fontsize=14)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(matrix_sizes)

for size in matrix_sizes:
    subset = df[df['matrix_size'] == size]
    ax2.plot(subset['procs'], subset['speedup'],
             marker='s', linewidth=2, markersize=6,
             label=f'Матрица {size}x{size}')

ax2.plot(procs_list, procs_list, 'k--', label='Идеальное ускорение', alpha=0.7, linewidth=2)

ax2.set_xlabel('Количество процессов', fontsize=12)
ax2.set_ylabel('Ускорение', fontsize=12)
ax2.set_title('Ускорение алгоритма Кэннона', fontsize=14)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(procs_list)

plt.tight_layout()
plt.savefig('matrix_plots/summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ТАБЛИЦА РЕЗУЛЬТАТОВ АЛГОРИТМА КЭННОНА")
print("="*80)
print("Формат: время (ускорение)")
print("="*80)

header = "Размер матрицы".ljust(15)
for procs in procs_list:
    if procs == 1:
        header += " | 1 процесс".ljust(18)
    else:
        header += f" | {procs} процессов".ljust(18)
print(header)
print("-" * len(header))

for size in matrix_sizes:
    subset = df[df['matrix_size'] == size].sort_values('procs')
    
    row = f"{size}x{size}".ljust(15)
    
    for procs in procs_list:
        data = subset[subset['procs'] == procs]
        if not data.empty:
            time_val = data['time'].values[0]
            speedup_val = data['speedup'].values[0]
            
            if procs == 1:
                cell = f"{time_val:.4f}".ljust(18)
            else:
                cell = f"{time_val:.4f} ({speedup_val:.2f})".ljust(18)
        else:
            cell = " ".ljust(18)
        
        row += " | " + cell
    
    print(row)
    print("-" * len(header))

print("\n" + "="*50)
print("АНАЛИЗ ЭФФЕКТИВНОСТИ")
print("="*50)

print("Эффективность параллелизма:")
for size in matrix_sizes:
    subset = df[df['matrix_size'] == size]
    max_procs = max(subset['procs'])
    if max_procs > 1:
        speedup_max = subset[subset['procs'] == max_procs]['speedup'].values[0]
        ideal = max_procs
        efficiency = (speedup_max / ideal) * 100
        print(f"• Матрица {size}x{size}: ускорение на {max_procs} процессах = {speedup_max:.2f} (эффективность: {efficiency:.1f}%)")

print(f"\nГрафик сохранен: 'matrix_plots/summary.png'")
print("Таблица результатов выведена выше")