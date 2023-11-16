#include "print.h"


void fprint_mat(std::string fname, Mat& matrix, int precision) {
    std::ofstream fout(fname);
    fout.precision(precision);
    fout << '[';
    for (auto& row: matrix) {
        fout << '[';
        for (double elem: row) {
            fout << elem << ',';
        }
        fout << "]," << std::endl;
    }
    fout << ']' << std::endl;
    fout.close();
}
