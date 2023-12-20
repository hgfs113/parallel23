#include <vector>
#include "backend.h"
#include <mpi.h>


Mat optimize(
        Backend &backend,
        int worker_id,
        int num_workers,
        double delta=1e-6,
        int max_iter=10000,
        bool verbose=false
);

void communicate(Mat obj, Mat buf, MPI_Status* status, int M, int N, int worker_id, int grid_size, int num_workers);