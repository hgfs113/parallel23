#include <iostream>
#include <cmath>
#include "omp.h"
#include "print.h"
#include "opt.h"
#include <mpi.h>

int num_workers, worker_id;

double solve(int M, int N) {
    double start = omp_get_wtime();
    Backend backend(M, N);
    Mat w_opt = optimize(backend, worker_id, num_workers);
    double end = omp_get_wtime();
    fprint_mat("out.txt", w_opt, M, N, 8);
    return end - start;
}

int main(int argc, char **argv) {
    std::cout.precision(3);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_workers);
    MPI_Comm_rank(MPI_COMM_WORLD, &worker_id);

    int num_threads = 4;
    int M = 32, N = 32;
    omp_set_num_threads(num_threads);
    double time = solve(M, N);

    std::cout << "Time: " << time << std::endl;
    std::cout << "NumThreads: " << num_threads  << std::endl;
    std::cout << "Grid: " << M << "x" << N << std::endl;
    
    MPI_Finalize();
    return 0;
}