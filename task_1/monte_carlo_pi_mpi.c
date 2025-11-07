#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

long long monte_carlo_pi_local(long long num_points, int rank) {
    unsigned int seed = (unsigned int)time(NULL) + rank;
    long long local_count = 0;
    
    for (long long i = 0; i < num_points; i++) {
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0; // [-1, 1]
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0; // [-1, 1]
        
        if (x*x + y*y <= 1.0) {
            local_count++;
        }
    }
    
    return local_count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    long long total_points;
    int method = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 3) {
        if (rank == 0) {
            printf("Использование: %s <общее_количество_точек> <метод>\n", argv[0]);
            printf("Методы: 0 - равномерное распределение\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    total_points = atoll(argv[1]);
    method = atoi(argv[2]);
    
    if (total_points <= 0) {
        if (rank == 0) {
            printf("Ошибка: количество точек должно быть положительным\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    srand((unsigned int)time(NULL) + rank);
    
    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    long long local_points, local_count;
    long long total_count = 0;
    
    if (method == 0) {
        local_points = total_points / size;
        if (rank < total_points % size) {
            local_points++;
        }
    } else {
        local_points = total_points / size;
    }
    
    local_count = monte_carlo_pi_local(local_points, rank);
    
    MPI_Reduce(&local_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        double pi_estimate = 4.0 * (double)total_count / (double)total_points;
        double error = fabs(pi_estimate - M_PI);
        double execution_time = end_time - start_time;
        
        printf("Метод: %s\n", (method == 0) ? "равномерное" : "блочное");
        printf("Количество процессов: %d\n", size);
        printf("Общее количество точек: %lld\n", total_points);
        printf("Вычисленное значение π: %.10f\n", pi_estimate);
        printf("Точное значение π: %.10f\n", M_PI);
        printf("Погрешность: %.10f\n", error);
        printf("Время выполнения: %.6f секунд\n", execution_time);
        printf("Точек в секунду: %.0f\n", total_points / execution_time);
        
        FILE *csv_file = fopen("results.csv", "a");
        if (csv_file != NULL) {
            fprintf(csv_file, "%d,%lld,%d,%.6f,%.10f,%.10f\n", 
                    size, total_points, method, execution_time, pi_estimate, error);
            fclose(csv_file);
        }
    }
    
    MPI_Finalize();
    return 0;
}
