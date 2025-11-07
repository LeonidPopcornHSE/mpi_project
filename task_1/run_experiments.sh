#!/bin/bash

echo "Компиляция monte_carlo_pi_mpi.c..."
mpicc -O3 -std=c99 -o monte_carlo_pi_mpi monte_carlo_pi_mpi.c -lm

echo "procs,points,method,time,pi_estimate,error" > results.csv

PROCS=(1 2 3 4)
POINTS=(1000000 5000000 10000000 50000000 100000000)
METHODS=(0)

echo "Запуск экспериментов..."

for method in "${METHODS[@]}"; do
    for points in "${POINTS[@]}"; do
        for procs in "${PROCS[@]}"; do
            echo "Запуск: mpirun -np $procs ./monte_carlo_pi_mpi $points $method"
            
            output=$(mpirun -np $procs ./monte_carlo_pi_mpi $points $method 2>&1)
            echo "$output"
            
            time=$(echo "$output" | grep "Время выполнения:" | awk '{print $3}')
            pi_estimate=$(echo "$output" | grep "Вычисленное значение π:" | awk '{print $4}')
            error=$(echo "$output" | grep "Погрешность:" | awk '{print $2}')
            
            if [ -n "$time" ] && [ -n "$pi_estimate" ] && [ -n "$error" ]; then
                echo "$procs,$points,$method,$time,$pi_estimate,$error" >> results.csv
                echo "✓ Данные записаны в CSV"
            else
                echo "✗ Ошибка: не удалось извлечь данные из вывода"
            fi
            
            echo "---"
        done
    done
done

echo "Все эксперименты завершены!"
echo "Результаты сохранены в results.csv"
echo "Количество записей в CSV: $(wc -l < results.csv)"
