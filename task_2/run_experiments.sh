#!/bin/bash
# Пример использования: ./run_experiments.sh
BIN=./matvec_mpi
OUT=results.csv
MODES=("row" "col" "block")
SIZES=(1000 2000 5000 10000)   # адаптируйте под вашу машину/память
PROCS=(1 2 4)          # набор чисел процессов
RUNS=5
 
echo "mode,N,procs,avg_time" > $OUT
 
for mode in "${MODES[@]}"; do
  for N in "${SIZES[@]}"; do
    for p in "${PROCS[@]}"; do
      echo "Running mode=$mode N=$N p=$p ..."
      mpirun -np $p $BIN $mode $N $RUNS >> $OUT
      sleep 0.5
    done
  done
done
echo "Done. Results in $OUT"
