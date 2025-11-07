#!/bin/bash

echo "Компиляция matrix_mult_cannon.c..."
mpicc -O3 -std=c99 -o matrix_mult_cannon matrix_mult_cannon.c -lm

if [ $? -ne 0 ]; then
    echo "Ошибка компиляции!"
    exit 1
fi

echo "procs,matrix_size,grid_size,time" > matrix_results.csv

PROCS=(1 4)
MATRIX_SIZES=(300 600 900 1500 2000)

echo "Запуск экспериментов с умножением матриц..."

for matrix_size in "${MATRIX_SIZES[@]}"; do
    for procs in "${PROCS[@]}"; do
        case $procs in
            1) grid_size=1 ;;
            4) grid_size=2 ;;
            9) grid_size=3 ;;
            16) grid_size=4 ;;
            *) continue ;;
        esac
        
        if [ $((matrix_size % grid_size)) -ne 0 ]; then
            echo "Пропуск: размер матрицы $matrix_size не кратен размеру сетки $grid_size"
            continue
        fi
        
        echo "Запуск: mpirun -np $procs ./matrix_mult_cannon $matrix_size"
        
        timeout 60s mpirun -np $procs ./matrix_mult_cannon $matrix_size 2>&1
        result=$?
        
        if [ $result -eq 0 ]; then
            echo "✓ Успешно завершено"
        elif [ $result -eq 248 ]; then
            echo "✗ Таймаут: программа зависла"
        else
            echo "✗ Ошибка выполнения: код $result"
        fi
        
        echo "---"
    done
done

echo "Все эксперименты завершены!"
