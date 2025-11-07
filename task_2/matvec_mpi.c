#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
 
void die(const char *s){ fprintf(stderr,"%s\n",s); MPI_Abort(MPI_COMM_WORLD,1); }
 
void block_range_1d(int rank, int p, int N, int *start, int *count) {
    int base = N / p;
    int rem = N % p;
    if (rank < rem) {
        *start = rank * (base + 1);
        *count = base + 1;
    } else {
        *start = rem * (base + 1) + (rank - rem) * base;
        *count = base;
    }
}
 
double *alloc_double(size_t n) {
    double *p = (double*)malloc(sizeof(double)*n);
    if (!p) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    return p;
}
 
void fill_random(double *a, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i=0;i<n;i++) a[i] = ((double)rand() / RAND_MAX);
}
 
void matvec_local(double *A, int rows, int cols, double *x_local, double *y_part) {
    for (int i=0;i<rows;i++) {
        double s = 0.0;
        double *Ai = A + (size_t)i * cols;
        for (int j=0;j<cols;j++) s += Ai[j] * x_local[j];
        y_part[i] = s;
    }
}
 
int main(int argc, char **argv){
    MPI_Init(&argc,&argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&p);
 
    if (argc < 4) {
        if (rank==0) fprintf(stderr,"Usage: %s {row|col|block} N runs\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
 
    const char *mode = argv[1];
    int N = atoi(argv[2]);
    int runs = atoi(argv[3]);
    if (N<=0 || runs<=0) die("N and runs must be > 0");
 
    double *A_full = NULL;
    double *x_full = NULL;
    double *y_full = NULL;
    unsigned int seed = (unsigned int)time(NULL) + rank*97;
    double total_time = 0.0;
 
    if (strcmp(mode,"row")==0) {
        int start, local_rows;
        block_range_1d(rank,p,N,&start,&local_rows);
        double *A_local = alloc_double((size_t)local_rows * N);
        double *x = alloc_double(N);
        double *y_local = alloc_double(local_rows);
        int *sendcounts = NULL, *displs = NULL;
        if (rank==0) {
            A_full = alloc_double((size_t)N*N);
            x_full = alloc_double(N);
            fill_random(A_full, (size_t)N*N, seed + 1234);
            fill_random(x_full, (size_t)N, seed + 4321);
            sendcounts = (int*)malloc(sizeof(int)*p);
            displs = (int*)malloc(sizeof(int)*p);
            for (int r=0;r<p;r++){
                int s, cnt;
                block_range_1d(r,p,N,&s,&cnt);
                sendcounts[r] = cnt * N;
                displs[r] = s * N;
            }
        }
        MPI_Scatterv(A_full, sendcounts, displs, MPI_DOUBLE,
                     A_local, local_rows * N, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        if (rank==0) memcpy(x, x_full, sizeof(double)*N);
        MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int t=0;t<runs;t++){
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            matvec_local(A_local, local_rows, N, x, y_local);
            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            double elapsed = t1 - t0;
            double max_elapsed;
            MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank==0) total_time += max_elapsed;
        }
        int *recvcounts = NULL, *rdispls = NULL;
        if (rank==0){
            y_full = alloc_double(N);
            recvcounts = (int*)malloc(sizeof(int)*p);
            rdispls = (int*)malloc(sizeof(int)*p);
            for (int r=0;r<p;r++){
                int s,c; block_range_1d(r,p,N,&s,&c);
                recvcounts[r] = c;
                rdispls[r] = s;
            }
        }
        MPI_Gatherv(y_local, local_rows, MPI_DOUBLE,
                    y_full, recvcounts, rdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        free(A_local); free(x); free(y_local);
        if (rank==0){
            free(A_full); free(x_full); free(y_full);
            free(sendcounts); free(displs); free(recvcounts); free(rdispls);
        }
    } else if (strcmp(mode,"col")==0) {
        int start_col, local_cols;
        block_range_1d(rank,p,N,&start_col,&local_cols);
        double *A_local = alloc_double((size_t)N * local_cols);
        double *x_local = alloc_double(local_cols);
        double *y_partial = alloc_double(N);
        if (rank==0){
            A_full = alloc_double((size_t)N * N);
            x_full = alloc_double(N);
            fill_random(A_full, (size_t)N*N, seed + 1234);
            fill_random(x_full, (size_t)N, seed + 4321);
        }
        if (rank==0) {
            for (int r=0;r<p;r++){
                int s,c; block_range_1d(r,p,N,&s,&c);
                if (r==0) {
                    for (int i=0;i<N;i++){
                        for (int j=0;j<c;j++){
                            A_local[(size_t)i * local_cols + j] = A_full[(size_t)i * N + (s + j)];
                        }
                    }
                } else {
                    double *buf = (double*)malloc(sizeof(double) * (size_t)N * c);
                    for (int i=0;i<N;i++){
                        for (int j=0;j<c;j++){
                            buf[(size_t)i * c + j] = A_full[(size_t)i * N + (s + j)];
                        }
                    }
                    MPI_Send(buf, N*c, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                    free(buf);
                }
            }
        } else {
            MPI_Recv(A_local, N*local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank==0) {
            for (int r=0;r<p;r++){
                int s,c; block_range_1d(r,p,N,&s,&c);
                if (r==0) {
                    memcpy(x_local, x_full + s, sizeof(double)*c);
                } else {
                    MPI_Send(x_full + s, c, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(x_local, local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i=0;i<N;i++) y_partial[i] = 0.0;
        for (int t=0;t<runs;t++){
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            for (int i=0;i<N;i++){
                double s = 0.0;
                double *Ai = A_local + (size_t)i * local_cols;
                for (int j=0;j<local_cols;j++) s += Ai[j] * x_local[j];
                y_partial[i] = s;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            double elapsed = t1 - t0;
            double max_elapsed;
            MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank==0) total_time += max_elapsed;
        }
        if (rank==0) {
            y_full = alloc_double(N);
        }
        MPI_Reduce(y_partial, y_full, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        free(A_local); free(x_local); free(y_partial);
        if (rank==0) { free(A_full); free(x_full); free(y_full); }
    } else if (strcmp(mode,"block")==0) {
        int dims[2] = {0,0};
        MPI_Dims_create(p,2,dims);
        int periods[2] = {0,0};
        MPI_Comm cart;
        MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,1,&cart);
        int coords[2];
        MPI_Cart_coords(cart, rank, 2, coords);
        int prow = dims[0], pcol = dims[1];
        int row_id = coords[0], col_id = coords[1];
        int row_start, local_rows;
        int col_start, local_cols;
        block_range_1d(row_id, prow, N, &row_start, &local_rows);
        block_range_1d(col_id, pcol, N, &col_start, &local_cols);
        double *A_local = alloc_double((size_t)local_rows * local_cols);
        double *x_local = alloc_double(local_cols);
        double *y_partial = alloc_double(local_rows);
        double *y_row_root = NULL;
        if (rank==0) {
            A_full = alloc_double((size_t)N*N);
            x_full = alloc_double(N);
            fill_random(A_full, (size_t)N*N, seed + 1234);
            fill_random(x_full, (size_t)N, seed + 4321);
        }
        if (rank==0) {
            for (int r=0;r<p;r++){
                int crd[2];
                MPI_Cart_coords(cart, r, 2, crd);
                int rrow = crd[0], rcol = crd[1];
                int r_row_start, r_local_rows;
                int r_col_start, r_local_cols;
                block_range_1d(rrow, prow, N, &r_row_start, &r_local_rows);
                block_range_1d(rcol, pcol, N, &r_col_start, &r_local_cols);
                if (r==0) {
                    for (int i=0;i<r_local_rows;i++){
                        for (int j=0;j<r_local_cols;j++){
                            A_local[(size_t)i * r_local_cols + j] = A_full[(size_t)(r_row_start + i) * N + (r_col_start + j)];
                        }
                    }
                } else {
                    double *buf = (double*)malloc(sizeof(double) * (size_t)r_local_rows * r_local_cols);
                    for (int i=0;i<r_local_rows;i++){
                        for (int j=0;j<r_local_cols;j++){
                            buf[(size_t)i * r_local_cols + j] = A_full[(size_t)(r_row_start + i) * N + (r_col_start + j)];
                        }
                    }
                    MPI_Send(buf, r_local_rows * r_local_cols, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                    free(buf);
                }
            }
        } else {
            MPI_Recv(A_local, local_rows * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank==0) {
            for (int r=0;r<p;r++){
                int crd[2];
                MPI_Cart_coords(cart, r, 2, crd);
                int rcol = crd[1];
                int r_col_start, r_local_cols;
                block_range_1d(rcol, pcol, N, &r_col_start, &r_local_cols);
                if (r==0) {
                    memcpy(x_local, x_full + r_col_start, sizeof(double)*r_local_cols);
                } else {
                    MPI_Send(x_full + r_col_start, r_local_cols, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(x_local, local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int t=0;t<runs;t++){
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            for (int i=0;i<local_rows;i++){
                double s = 0.0;
                double *Ai = A_local + (size_t)i * local_cols;
                for (int j=0;j<local_cols;j++) s += Ai[j] * x_local[j];
                y_partial[i] = s;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            double elapsed = t1 - t0;
            double max_elapsed;
            MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank==0) total_time += max_elapsed;
        }
        int color_row = row_id;
        MPI_Comm row_comm;
        MPI_Comm_split(MPI_COMM_WORLD, color_row, col_id, &row_comm);
        int row_rank, row_size;
        MPI_Comm_rank(row_comm, &row_rank);
        MPI_Comm_size(row_comm, &row_size);
        if (col_id == 0) {
            y_row_root = alloc_double(local_rows);
        }
        MPI_Reduce(y_partial, y_row_root, local_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);
        if (col_id == 0) {
            if (rank == 0) {
                y_full = alloc_double(N);
                memcpy(y_full + row_start, y_row_root, sizeof(double)*local_rows);
                for (int r=0;r<prow;r++){
                    if (r==row_id) continue;
                    int coords2[2] = {r,0};
                    int dest_rank;
                    MPI_Cart_rank(cart, coords2, &dest_rank);
                    int r_row_start, r_local_rows;
                    block_range_1d(r, prow, N, &r_row_start, &r_local_rows);
                    MPI_Recv(y_full + r_row_start, r_local_rows, MPI_DOUBLE, dest_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            } else {
                int dest = 0;
                MPI_Send(y_row_root, local_rows, MPI_DOUBLE, dest, 2, MPI_COMM_WORLD);
            }
        }
        free(A_local); free(x_local); free(y_partial);
        if (y_row_root) free(y_row_root);
        if (rank==0) {
            if (A_full) free(A_full);
            if (x_full) free(x_full);
            if (y_full) free(y_full);
        }
        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&cart);
    } else {
        if (rank==0) fprintf(stderr,"Unknown mode '%s'\n", mode);
        MPI_Finalize();
        return 1;
    }
 
    if (rank==0) {
        double avg = total_time / runs;
        printf("%s,%d,%d,%.9f\n", mode, N, p, avg);
        fflush(stdout);
    }
 
    MPI_Finalize();
    return 0;
}
