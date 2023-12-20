#include "print.h"


void fprint_mat(std::string fname, Mat& matrix, int M, int N, int precision) {
    std::ofstream fout(fname);
    fout.precision(precision);
    fout << '[';
    for (int i = 0; i < M + 1; ++i) {
        fout << '[';
        for (int j = 0; j < N + 1; ++j) {
            fout << matrix[i * (N + 1) + j] << ',';
        }
        fout << "]," << std::endl;
    }
    fout << ']' << std::endl;
    fout.close();
}
