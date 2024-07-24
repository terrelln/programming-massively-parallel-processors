#pragma once

#include <cstddef>

namespace terrelln {

void matrixMultiplyByRow(float *P_h, float const *M_h, float const *N_h,
                         size_t M_rows, size_t M_cols, size_t N_rows,
                         size_t N_cols, size_t reps);

void matrixMultiplyByCol(float *P_h, float const *M_h, float const *N_h,
                         size_t M_rows, size_t M_cols, size_t N_rows,
                         size_t N_cols, size_t reps);

} // namespace terrelln