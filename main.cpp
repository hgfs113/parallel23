#include <iostream>
#include <cmath>
#include "/opt/homebrew/opt/libomp/include/omp.h"
#include "print.h"
#include "opt.h"


double solve(int M, int N) {
    double start = omp_get_wtime();
    Backend backend(M, N);
    Mat w_opt = optimize(backend);
    double end = omp_get_wtime();
    fprint_mat("out.txt", w_opt, 8);
    return end - start;
}


int main() {
    std::cout.precision(3);

    int num_threads = 20;
    int M = 32, N = 32;
    omp_set_num_threads(num_threads);
    double time = solve(M, N);

    std::cout << "Time: " << time << std::endl;
    std::cout << "NumThreads: " << num_threads  << std::endl;
    std::cout << "Grid: " << M << "x" << N << std::endl;
    
    return 0;
}