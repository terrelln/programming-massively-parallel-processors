#include "Exercise1.hpp"

#include <stdexcept>

namespace terrelln {
namespace {
void checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA Runtime Error: ") +
                             cudaGetErrorString(result));
  }
}
} // namespace

__global__ void matrixMultiplyByRowKernel(float *P, float const *M,
                                          float const *N, size_t P_rows,
                                          size_t P_cols, size_t MN_inner) {
  auto const rows = P_rows;
  size_t const col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < P_cols) {
    for (size_t row = 0; row < rows; ++row) {
      float sum = 0;
      for (size_t k = 0; k < MN_inner; ++k) {
        // M reads are not contiguous by thread
        // N reads are not contiguous by thread
        sum += M[k * P_rows + row] * N[col * MN_inner + k];
      }
      // P writes are not contiguous by thread
      P[col * P_rows + row] = sum;
    }
  }
}

void matrixMultiplyByRow(float *P_h, float const *M_h, float const *N_h,
                         size_t M_rows, size_t M_cols, size_t N_rows,
                         size_t N_cols, size_t reps) {
  auto const rows = M_rows;
  auto const cols = N_cols;
  auto const inner = M_cols;
  if (M_cols != N_rows) {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  float *P_d;
  float *M_d;
  float *N_d;

  size_t const P_bytes = rows * cols * sizeof(float);
  size_t const M_bytes = rows * inner * sizeof(float);
  size_t const N_bytes = cols * inner * sizeof(float);

  checkCuda(cudaMalloc(&P_d, P_bytes));
  checkCuda(cudaMalloc(&M_d, M_bytes));
  checkCuda(cudaMalloc(&N_d, N_bytes));

  checkCuda(cudaMemcpy(M_d, M_h, M_bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(N_d, N_h, N_bytes, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < reps; ++i) {
    auto const kThreadsPerBlock = 128;
    auto const blocks = (cols + kThreadsPerBlock - 1) / kThreadsPerBlock;
    matrixMultiplyByRowKernel<<<blocks, kThreadsPerBlock>>>(P_d, M_d, N_d, rows,
                                                            cols, inner);
  }

  checkCuda(cudaMemcpy(P_h, P_d, P_bytes, cudaMemcpyDeviceToHost));

  checkCuda(cudaFree(P_d));
  checkCuda(cudaFree(M_d));
  checkCuda(cudaFree(N_d));
}

__global__ void matrixMultiplyByColKernel(float *P, float const *M,
                                          float const *N, size_t P_rows,
                                          size_t P_cols, size_t MN_inner) {

  size_t const row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < P_rows) {
    for (size_t col = 0; col < P_cols; ++col) {
      float sum = 0;
      for (size_t k = 0; k < MN_inner; ++k) {
        // M reads are contiguous by thread
        // N reads are not contiguous by thread
        sum += M[k * P_rows + row] * N[col * MN_inner + k];
      }
      // P writes are contiguous by thread
      P[col * P_rows + row] = sum;
    }
  }
}

void matrixMultiplyByCol(float *P_h, float const *M_h, float const *N_h,
                         size_t M_rows, size_t M_cols, size_t N_rows,
                         size_t N_cols, size_t reps) {
  auto const rows = M_rows;
  auto const cols = N_cols;
  auto const inner = M_cols;
  if (M_cols != N_rows) {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  float *P_d;
  float *M_d;
  float *N_d;

  size_t const P_bytes = rows * cols * sizeof(float);
  size_t const M_bytes = rows * inner * sizeof(float);
  size_t const N_bytes = cols * inner * sizeof(float);

  checkCuda(cudaMalloc(&P_d, P_bytes));
  checkCuda(cudaMalloc(&M_d, M_bytes));
  checkCuda(cudaMalloc(&N_d, N_bytes));

  checkCuda(cudaMemcpy(M_d, M_h, M_bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(N_d, N_h, N_bytes, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < reps; ++i) {
    auto const kThreadsPerBlock = 128;
    auto const blocks = (rows + kThreadsPerBlock - 1) / kThreadsPerBlock;
    matrixMultiplyByColKernel<<<blocks, kThreadsPerBlock>>>(P_d, M_d, N_d, rows,
                                                            cols, inner);
  }

  checkCuda(cudaMemcpy(P_h, P_d, P_bytes, cudaMemcpyDeviceToHost));

  checkCuda(cudaFree(P_d));
  checkCuda(cudaFree(M_d));
  checkCuda(cudaFree(N_d));
}

} // namespace terrelln