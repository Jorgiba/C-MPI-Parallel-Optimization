#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double compute(const double *a, int n)
{
    double acc = 0.0;
    for (int i = 0; i < n; ++i)
        acc += a[i] * a[i];
    return acc;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long N = (argc > 1) ? atol(argv[1]) : 1000000L;

    double start_time, end_time, computation_time;

    int *counts = NULL, *displs = NULL;
    double *A = NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    if (rank == 0)
    {
        counts = (int *)malloc(sizeof(int) * size);
        displs = (int *)malloc(sizeof(int) * size);
        long base = N / size, rem = N % size, off = 0;
        for (int r = 0; r < size; ++r)
        {
            long c = base + (r < rem);
            counts[r] = (int)c;
            displs[r] = (int)off;
            off += c;
        }
        A = (double *)malloc(sizeof(double) * N);
        srand(42); 
        for (long i = 0; i < N; ++i)
            A[i] = (double)rand() / RAND_MAX;
    }

    int local_n = 0;
    MPI_Scatter(counts, 1, MPI_INT, &local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *chunk = local_n ? (double *)malloc(sizeof(double) * local_n) : NULL;
    MPI_Scatterv(A, counts, displs, MPI_DOUBLE, chunk, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double comp_start = MPI_Wtime();
    double partial = local_n ? compute(chunk, local_n) : 0.0;
    double comp_end = MPI_Wtime();
    computation_time = comp_end - comp_start;

    double total = 0.0;
    MPI_Reduce(&partial, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    if (rank == 0)
    {
        printf("RESULTADO: N=%ld, Procesos=%d, TiempoTotal=%.6f, TiempoComp=%.6f, SumaCuadrados=%.12f\n",
               N, size, total_time, computation_time, total);
    }

    free(chunk);
    if (rank == 0)
    {
        free(A);
        free(counts);
        free(displs);
    }
    MPI_Finalize();
    return 0;
}
