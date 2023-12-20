#include <vector>
#include <cmath>
#include <iostream>


typedef std::vector<double> Mat;


class Backend {
    int M, N;

    double A1 = 0.0;
    double B1 = 3.0;
    double A2 = 0.0;
    double B2 = 4.0;

    double h1, h2;
    double h1h2;
    double eps;

    Mat a, b, B;

    bool is_almost_eq_f(double x, double y, double EPSILON = 1e-9){
        return fabs(x - y) < EPSILON;
    }

    bool is_inside_figure(double x, double y) {
        double in = x + 0.75 * y - 3;
        return in < 0 or is_almost_eq_f(in, 0.0);
    }

    double clip_x(double x) {
        return fmin(B1, fmax(A1, x));
    }

    double clip_y(double y) {
        return fmin(B2, fmax(A2, y));
    }

    double get_a(int i, int j);
    double get_b(int i, int j);
    double get_Fij(int i, int j);

public:
    Backend(int M, int N);

    void compute_A(int i_low, int i_high, int j_low, int j_high, Mat &w, Mat& Aw);

    int get_M() const {
        return M;
    }

    int get_N() const {
        return N;
    }

    const Mat& get_B() const {
        return B;
    }
};
