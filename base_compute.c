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

    int *counts = NULL, *displs = NULL;
    double *A = NULL;

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
        {
            A[i] = (double)rand() / RAND_MAX;
            printf("%f \n", A[i]);
        }
        for (int r = 1; r < size; ++r)
        {
            if (counts[r] > 0)
            {
                MPI_Send(A + displs[r], counts[r], MPI_DOUBLE, r, 11, MPI_COMM_WORLD);
            }
            else
            {
                MPI_Send(A, 0, MPI_DOUBLE, r, 11, MPI_COMM_WORLD);
            }
        }
    }

    long base = N / size, rem = N % size;
    int local_n = (int)(base + (rank < rem));
    double *chunk = local_n ? (double *)malloc(sizeof(double) * local_n) : NULL;

    if (rank == 0)
    {
        for (int i = 0; i < local_n; ++i)
            chunk[i] = A[displs[0] + i];
    }
    else
    {
        if (local_n > 0)
            MPI_Recv(chunk, local_n, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double partial = local_n ? compute(chunk, local_n) : 0.0;

    if (rank == 0)
    {
        double total = partial;
        for (int r = 1; r < size; ++r)
        {
            double tmp = 0.0;
            MPI_Recv(&tmp, 1, MPI_DOUBLE, r, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += tmp;
        }
        printf("N=%ld, size=%d, sumsq=%.12f\n", N, size, total);
    }
    else
    {
        MPI_Send(&partial, 1, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD);
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
