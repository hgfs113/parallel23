#include "opt.h"


Mat optimize(Backend &backend, double delta, int max_iter, bool verbose) {
    int M = backend.get_M();
    int N = backend.get_N();

    Mat w_opt(M + 1, std::vector<double>(N + 1));
    Mat r(M + 1, std::vector<double>(N + 1));

    double current_upd = 0.0;
    double Tau, denom;
    int i, j;

    for(int iter = 0; iter < max_iter; ++iter) {
        // find residual
        const Mat& Aw = backend.compute_A(w_opt);
        const Mat& B = backend.get_B();

        #pragma omp parallel for private(i) private(j)
        for (i = 0; i < M + 1; ++i) {
            for (j = 0; j < N + 1; ++j) {
                r[i][j] = Aw[i][j] - B[i][j];
            }
        }

        // iteration parameter tau
        Tau = 0.0;
        denom = 0.0;
        Mat Ar = backend.compute_A(r);

        #pragma omp parallel for private(i) private(j) reduction(+: Tau)
        for (i = 0; i < M + 1; ++i) {
            for (j = 0; j < N + 1; ++j) {
                Tau += Ar[i][j] * r[i][j];
            }
        }
        #pragma omp parallel for private(i) private(j) reduction(+: denom)
        for (i = 0; i < M + 1; ++i) {
            for (j = 0; j < N + 1; ++j) {
                denom += Ar[i][j] * Ar[i][j];
            }
        }
        Tau /= denom;

        // update step
        #pragma omp parallel for private(i) private(j)
        for (i = 0; i < M + 1; ++i) {
            for (j = 0; j < N + 1; ++j) {
                w_opt[i][j] -= Tau * r[i][j];
            }
        }

        // corner conditions
        #pragma omp parallel for private(i)
        for (i = 0; i < M + 1; ++i) {
            w_opt[i][0] = 0;
            w_opt[i][N] = 0;
        }
        #pragma omp parallel for private(j)
        for (j = 0; j < N + 1; ++j) {
            w_opt[0][j] = 0;
            w_opt[M][j] = 0;
        }

        // check stop rule
        double update_maxnorm = 0.0;
        #pragma omp parallel for private(i) private(j) reduction(max: update_maxnorm)
        for (i = 0; i < M + 1; ++i) {
            for (j = 0; j < N + 1; ++j) {
                update_maxnorm = fmax(update_maxnorm, fabs(r[i][j]));
            }
        }
        update_maxnorm = fabs(update_maxnorm * Tau);

        if (verbose) {
            std::cerr <<  "Iter[" << iter << "/" << max_iter << "] update_maxnorm = "<< update_maxnorm << std::endl;
        }
        
        if (update_maxnorm < delta) {
            return w_opt;
        }
    }

    if (verbose) {
        std::cerr <<  "WARNING! Method has not converged" << std::endl;
    }
    return w_opt;
}
