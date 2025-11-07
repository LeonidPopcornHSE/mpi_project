#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

void initialize_matrix(double *matrix, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 10.0;
    }
}

void matrix_multiply(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int grid_size;
    int matrix_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank == 0) {
            printf("Использование: %s <размер_матрицы>\n", argv[0]);
            printf("Количество процессов должно быть квадратом целого числа (1, 4, 9, 16, ...)\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    matrix_size = atoi(argv[1]);
    
    grid_size = 1;
    while (grid_size * grid_size < size) {
        grid_size++;
    }
    
    if (grid_size * grid_size != size) {
        if (rank == 0) {
            printf("Ошибка: количество процессов должно быть квадратом целого числа\n");
            printf("Текущее количество процессов: %d (допустимые значения: 1, 4, 9, 16, ...)\n", size);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (matrix_size % grid_size != 0) {
        if (rank == 0) {
            printf("Ошибка: размер матрицы должен быть кратен размеру сетки процессов\n");
            printf("Размер матрицы: %d, размер сетки: %d\n", matrix_size, grid_size);
        }
        MPI_Finalize();
        return 1;
    }
    
    int block_size = matrix_size / grid_size;
    
    if (rank == 0) {
        printf("=== Алгоритм Кэннона ===\n");
        printf("Количество процессов: %d (сетка %dx%d)\n", size, grid_size, grid_size);
        printf("Размер матрицы: %dx%d\n", matrix_size, matrix_size);
        printf("Размер блока: %dx%d\n", block_size, block_size);
    }
    
    int dims[2] = {grid_size, grid_size};
    int periods[2] = {1, 1};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];
    
    double *local_A = (double*)malloc(block_size * block_size * sizeof(double));
    double *local_B = (double*)malloc(block_size * block_size * sizeof(double));
    double *local_C = (double*)calloc(block_size * block_size, sizeof(double));
    double *temp_buffer = (double*)malloc(block_size * block_size * sizeof(double));
    
    double *A = NULL, *B = NULL;
    if (rank == 0) {
        A = (double*)malloc(matrix_size * matrix_size * sizeof(double));
        B = (double*)malloc(matrix_size * matrix_size * sizeof(double));
        
        initialize_matrix(A, matrix_size, 1);
        initialize_matrix(B, matrix_size, 2);
        
        printf("Матрицы инициализированы\n");
    }
    
    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    if (rank == 0) {
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int dest_rank;
                int dest_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
                
                if (dest_rank == 0) {
                    for (int x = 0; x < block_size; x++) {
                        for (int y = 0; y < block_size; y++) {
                            local_A[x * block_size + y] = A[(i * block_size + x) * matrix_size + (j * block_size + y)];
                            local_B[x * block_size + y] = B[(i * block_size + x) * matrix_size + (j * block_size + y)];
                        }
                    }
                } else {
                    MPI_Send(&A[i * block_size * matrix_size + j * block_size], 
                            block_size * block_size, MPI_DOUBLE, dest_rank, 0, MPI_COMM_WORLD);
                    MPI_Send(&B[i * block_size * matrix_size + j * block_size], 
                            block_size * block_size, MPI_DOUBLE, dest_rank, 1, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        MPI_Recv(local_A, block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_B, block_size * block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if (rank == 0) {
        printf("Распределение данных завершено\n");
    }
    
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, row, col, &row_comm);
    MPI_Comm_split(grid_comm, col, row, &col_comm);
    
    int left_rank, right_rank, up_rank, down_rank;
    MPI_Cart_shift(grid_comm, 1, -1, &left_rank, &right_rank);
    MPI_Cart_shift(grid_comm, 0, -1, &up_rank, &down_rank);
    
    int temp_rank;
    MPI_Sendrecv(local_A, block_size * block_size, MPI_DOUBLE, left_rank, 0,
                 temp_buffer, block_size * block_size, MPI_DOUBLE, right_rank, 0,
                 grid_comm, MPI_STATUS_IGNORE);
     
    MPI_Sendrecv(local_B, block_size * block_size, MPI_DOUBLE, up_rank, 0,
                 temp_buffer, block_size * block_size, MPI_DOUBLE, down_rank, 0,
                 grid_comm, MPI_STATUS_IGNORE);

    for (int step = 0; step < grid_size; step++) {
        matrix_multiply(local_A, local_B, temp_buffer, block_size);
        
        for (int i = 0; i < block_size * block_size; i++) {
            local_C[i] += temp_buffer[i];
        }
        
        MPI_Sendrecv(local_A, block_size * block_size, MPI_DOUBLE, left_rank, 0,
                     temp_buffer, block_size * block_size, MPI_DOUBLE, right_rank, 0,
                     grid_comm, MPI_STATUS_IGNORE);
        memcpy(local_A, temp_buffer, block_size * block_size * sizeof(double));
        
        MPI_Sendrecv(local_B, block_size * block_size, MPI_DOUBLE, up_rank, 0,
                     temp_buffer, block_size * block_size, MPI_DOUBLE, down_rank, 0,
                     grid_comm, MPI_STATUS_IGNORE);
        memcpy(local_B, temp_buffer, block_size * block_size * sizeof(double));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    double execution_time = end_time - start_time;
    
    if (rank == 0) {
        printf("Время выполнения: %.6f секунд\n", execution_time);
        
        FILE *csv_file = fopen("matrix_results.csv", "a");
        if (csv_file != NULL) {
            fprintf(csv_file, "%d,%d,%d,%.6f\n", 
                    size, matrix_size, grid_size, execution_time);
            fclose(csv_file);
        }
        
        free(A);
        free(B);
    }
    
    free(local_A);
    free(local_B);
    free(local_C);
    free(temp_buffer);
    
    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    
    MPI_Finalize();
    return 0;
}