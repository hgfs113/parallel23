#include "backend.h"


double Backend::get_Fij(int i, int j) {
    double x1 = clip_x(A1 + (i - 0.5) * h1);
    double y1 = clip_y(A2 + (j - 0.5) * h2);
    bool chk1 = is_inside_figure(x1, y1);

    double x2 = clip_x(A1 + (i + 0.5) * h1);
    double y2 = clip_y(A2 + (j + 0.5) * h2);        
    bool chk2 = is_inside_figure(x2, y2);

    double result = 0.0;

    if (chk1 && chk2) {
        result = (x2 - x1) * (y2 - y1);
    } else if (chk1) {
        double x_int = -0.75 * y1 + 3;
        double y_int = -4 * x1 / 3.0 + 4;
        if (
            (x_int < x2 && y_int < y2) || (is_almost_eq_f(x_int, x2) && is_almost_eq_f(y_int, y2))
        ) {
            // lower triangle
            result = 0.5 * (x_int - x1) * (y_int - y1);
        } else if (y_int > y2 && x_int < x2) {
            // trapezoid
            double x_int_2 = -0.75 * y2 + 3;
            result = (x_int_2 - x1 + x_int - x1) * 0.5 * (y2 - y1);
        } else if (y_int > y2 && x_int > x2) {
            // upper triangle
            double x_int_2 = -0.75 * y2 + 3;
            double y_int_2 = -4 * x2 / 3.0 + 4;
            result =  0.5 * (y2 - y_int_2) * (x2 - x_int_2);
        } else {
            throw "WtF is happening?";
        }
    }

    return result / h1h2;
}


double Backend::get_a(int i, int j) {
    double y_low = clip_y(A2 + (j - 0.5) * h2);
    double y_high = clip_y(A2 + (j + 0.5) * h2);
    double x = clip_x(A1 + (i - 0.5) * h1);
    
    bool chk1 = is_inside_figure(x, y_low);
    bool chk2 = is_inside_figure(x, y_high);

    double result = 0.0;
    if (chk1 && chk2) {
        result = y_high - y_low;
    } else if (chk1) {
        double y_mid = 4 - 4 * x / 3.0;
        result = (y_mid - y_low) + (y_high - y_mid) / eps;
    } else {
        result = (y_high - y_low) / eps;
    }
    return result / h2;
}


double Backend::get_b(int i, int j) {
    double x_low = clip_x(A1 + (i - 0.5) * h1);
    double x_high = clip_x(A1 + (i + 0.5) * h1);
    double y = clip_y(A2 + (j - 0.5) * h2);

    bool chk1 = is_inside_figure(x_low, y);
    bool chk2 = is_inside_figure(x_high, y);

    double result = 0.0;
    if (chk1 && chk2) {
        result = x_high - x_low;
    } else if (chk1) {
        double x_mid = 3 - 0.75 * y;
        result = (x_mid - x_low) + (x_high - x_mid) / eps;
    } else {
        result = (x_high - x_low) / eps;
    }
    return result / h1;
}


Backend::Backend(int M, int N): M(M), N(N) {
    h1 = (B1 - A1) / M;
    h2 = (B2 - A2) / N;
    h1h2 = h1 * h2;

    eps = fmax(h1, h2);
    eps *= eps;

    a = Mat(M + 1, std::vector<double>(N + 1));
    b = Mat(M + 1, std::vector<double>(N + 1));

    int i, j;
    #pragma omp parallel for private(i) private(j)
    for (i = 0; i < M + 1; ++i) {
        for (j = 0; j < N + 1; ++j) {
            a[i][j] = get_a(i, j);
            b[i][j] = get_b(i, j);
        }
    }

    B = Mat(M + 1, std::vector<double>(N + 1));
    #pragma omp parallel for private(i) private(j)
    for (i = 0; i < M + 1; ++i) {
        for (j = 0; j < N + 1; ++j) {
            B[i][j] = get_Fij(i, j);
        }
    }
}


Mat Backend::compute_A(Mat &w){
    Mat Aw(M + 1, std::vector<double>(N + 1));
    int i, j;
    #pragma omp parallel for private(i) private(j)
    for (i = 1; i < M; ++i) {
        for (j = 1; j < N; ++j) {
            if (i > 0 && j > 0 && i < M && j < N) {
                Aw[i][j] = -1.0 / h1 * (
                    a[i + 1][j] * (w[i + 1][j] - w[i][j]) / h1 - a[i][j] * (w[i][j] - w[i - 1][j]) / h1
                ) - 1.0 / h2 * (
                    b[i][j + 1] * (w[i][j + 1] - w[i][j]) / h2 - b[i][j] * (w[i][j] - w[i][j - 1]) / h2
                );
            }
        }
    }
    return Aw;
}
