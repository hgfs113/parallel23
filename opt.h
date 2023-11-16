#include <vector>
#include "backend.h"


Mat optimize(
        Backend &backend,
        double delta=1e-6,
        int max_iter=10000,
        bool verbose=false
);
