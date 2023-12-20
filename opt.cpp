#include "opt.h"


void communicate(Mat w, Mat buf, MPI_Status* status, int M, int N, int worker_id, int grid_size, int num_workers) {
    if (num_workers == 1) {
        return;
    }
    if (worker_id == 0) {
        for (int id = 1; id < num_workers; ++id) {
            MPI_Recv(&buf[0], ((M + 1) * (N + 1)), MPI_DOUBLE, id, 0, MPI_COMM_WORLD, status);
            int i_grid, j_grid, i, j;

            int i_step = (M + 1) / grid_size;
            int j_step = (N + 1) / grid_size;

            for (i_grid = 0; i_grid < grid_size; ++i_grid) {
                for (j_grid = 0; j_grid < grid_size; ++j_grid) {
                    if ((i_grid * grid_size + j_grid) % num_workers == id) {

                        int i_low = i_step * i_grid;
                        int i_high = i_step * (i_grid + 1);
                        if (i_grid + 1 == grid_size) {
                            i_high = M + 1;
                        }
                        int j_low = j_step * j_grid;
                        int j_high = j_step * (j_grid + 1);
                        if (j_grid + 1 == grid_size) {
                            j_high = N + 1;
                        }

                        #pragma omp parallel for private(i) private(j)
                        for (i = i_low; i < i_high; ++i) {
                            for (j = j_low; j < j_high; ++j) {
                                w[i * (N + 1) + j] = buf[i * (N + 1) + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        MPI_Send(&w[0], ((M + 1) * (N + 1)), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&w[0], ((M + 1) * (N + 1)), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


Mat optimize(Backend &backend, int worker_id, int num_workers, double delta, int max_iter, bool verbose) {

    int grid_size = 1;
    while (grid_size * grid_size < num_workers) {
        ++grid_size;
    }

    int M = backend.get_M();
    int N = backend.get_N();

    Mat w_opt((M + 1) * (N + 1));
    Mat r((M + 1) * (N + 1));
    Mat buf((M + 1) * (N + 1));

    Mat Aw((M + 1) * (N + 1));
    Mat Ar((M + 1) * (N + 1));

    MPI_Status status;

    double current_upd = 0.0;
    double Tau, denom;
    int i_grid, j_grid, i, j;
    const Mat& B = backend.get_B();
    int i_step = (M + 1) / grid_size;
    int j_step = (N + 1) / grid_size;
    int i_low, i_high, j_low, j_high;

    for(int iter = 0; iter < max_iter; ++iter) {
        broadcast(w_opt, buf, &status, M, N, worker_id, grid_size, num_workers);

        // find residual
        for (i_grid = 0; i_grid < grid_size; ++i_grid) {
            for (j_grid = 0; j_grid < grid_size; ++j_grid) {
                if ((i_grid * grid_size + j_grid) % num_workers == worker_id) {
                    i_low = i_step * i_grid;
                    if (i_grid + 1 != grid_size) {
                        i_high = i_step * (i_grid + 1);
                    } else {
                        i_high = M + 1;
                    }
                    j_low = j_step * j_grid;
                    if (j_grid + 1 != grid_size) {
                        j_high = j_step * (j_grid + 1);
                    } else {
                        j_high = N + 1;
                    }
                    backend.compute_A(i_low, i_high, j_low, j_high, w_opt, Aw);
                    #pragma omp parallel for private(i) private(j)
                    for (i = i_low; i < i_high; ++i) {
                        for (j = j_low; j < j_high; ++j) {
                            r[i * (N + 1) + j] = Aw[i * (N + 1) + j] - B[i * (N + 1) + j];
                        }
                    }
                }
            }
        }
        broadcast(r, buf, &status, M, N, worker_id, grid_size, num_workers);

        // iteration parameter tau
        double Tau_local = 0.0;
        double denom_local = 0.0;

        for (i_grid = 0; i_grid < grid_size; ++i_grid) {
            for (j_grid = 0; j_grid < grid_size; ++j_grid) {
                if ((i_grid * grid_size + j_grid) % num_workers == worker_id) {
                    i_low = i_step * i_grid;
                    if (i_grid + 1 != grid_size) {
                        i_high = i_step * (i_grid + 1);
                    } else {
                        i_high = M + 1;
                    }
                    j_low = j_step * j_grid;
                    if (j_grid + 1 != grid_size) {
                        j_high = j_step * (j_grid + 1);
                    } else {
                        j_high = N + 1;
                    }
                    backend.compute_A(i_low, i_high, j_low, j_high, r, Ar);

                    double tau = 0.0;
                    #pragma omp parallel for private(i) private(j) reduction(+: tau)
                    for (i = i_low; i < i_high; ++i) {
                        for (j = j_low; j < j_high; ++j) {
                            tau += Ar[i * (N + 1) + j] * r[i * (N + 1) + j];
                        }
                    }

                    Tau_local += tau;

                    double denom = 0.0;
                    #pragma omp parallel for private(i) private(j) reduction(+: denom)
                    for (i = i_low; i < i_high; ++i) {
                        for (j = j_low; j < j_high; ++j) {
                            denom += Ar[i * (N + 1) + j] * Ar[i * (N + 1) + j];
                        }
                    }

                    denom_local += denom;
                }
            }
        }

        double Tau;
        MPI_Allreduce(&Tau_local, &Tau, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double Denom;
        MPI_Allreduce(&denom_local, &Denom, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        Tau /= Denom;


        double update_maxnorm_local = 0.0;
        for (i_grid = 0; i_grid < grid_size; ++i_grid) {
            for (j_grid = 0; j_grid < grid_size; ++j_grid) {
                if ((i_grid * grid_size + j_grid) % num_workers == worker_id) {
                    i_low = i_step * i_grid;
                    if (i_grid + 1 != grid_size) {
                        i_high = i_step * (i_grid + 1);
                    } else {
                        i_high = M + 1;
                    }
                    j_low = j_step * j_grid;
                    if (j_grid + 1 != grid_size) {
                        j_high = j_step * (j_grid + 1);
                    } else {
                        j_high = N + 1;
                    }

                    // update step
                    #pragma omp parallel for private(i) private(j)
                    for (i = i_low; i < i_high; ++i) {
                        for (j = j_low; j < j_high; ++j) {
                            w_opt[i * (N + 1) + j] -= Tau * r[i * (N + 1) + j];
                        }
                    }

                    // corner conditions
                    #pragma omp parallel for private(i)
                    for (i = i_low; i < i_high; ++i) {
                        w_opt[i * (N + 1)] = 0;
                        w_opt[i * (N + 1) + N] = 0;
                    }
                    #pragma omp parallel for private(j)
                    for (j = j_low; j < j_high; ++j) {
                        w_opt[j] = 0;
                        w_opt[M * (N + 1) + j] = 0;
                    }

                    // check stop rule
                    double update_maxnorm = 0.0;
                    #pragma omp parallel for private(i) private(j) reduction(max: update_maxnorm)
                    for (i = i_low; i < i_high; ++i) {
                        for (j = j_low; j < j_high; ++j) {
                            update_maxnorm = fmax(update_maxnorm, fabs(r[i * (N + 1) + j]));
                        }
                    }

                    update_maxnorm_local = fmax(update_maxnorm_local, update_maxnorm);

                }
            }
        }

        double Update_maxnorm;
        MPI_Allreduce(&update_maxnorm_local, &Update_maxnorm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        Update_maxnorm = fabs(Update_maxnorm * Tau);

        if (verbose && worker_id == 0) {
            std::cerr <<  "Iter[" << iter << "/" << max_iter << "] update_maxnorm = "<< Update_maxnorm << std::endl;
        }
        
        if (Update_maxnorm < delta) {
            return w_opt;
        }
    }

    if (verbose && worker_id == 0) {
        std::cerr <<  "WARNING! Method has not converged" << std::endl;
    }
    return w_opt;
}
