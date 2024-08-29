#include "chapter6.h"
#include <ATen/cuda/CUDAContext.h>

namespace {
constexpr size_t kTileSize = 32;

__global__ void exercise1(float *P, float const *M, float const *N,
                          size_t size) {
  __shared__ float Mds[kTileSize][kTileSize];
  __shared__ float Nds[kTileSize][kTileSize];

  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;

  int const row = by * kTileSize + ty;
  int const col = bx * kTileSize + tx;

  float Pvalue = 0.0f;
  for (int ph = 0; ph < size / kTileSize; ++ph) {
    Mds[ty][tx] = M[row * size + ph * kTileSize + tx];
    // Naive
    // Nds[ty][tx] = N[col * size + ph * kTileSize + ty];
    // "Optimized" but runs at exactly the same speed in my tests.
    // Maybe NVCC is smart enough to optimize this? I should check the generated code.
    Nds[tx][ty] = N[(bx * kTileSize + ty) * size + ph * kTileSize + tx];
    // Swapping ty & tx in the store and correspondingly in the access below is 2x as slower
    // Naively I thought it would be faster because we are both loading & storing contiguously,
    // or at least neutral.
    // but that makes the Nds local access non-contiguous, and I guess this load can still be
    // coalesced even if the writes are not.
    // Nds[ty][tx] = N[(bx * kTileSize + ty) * size + ph * kTileSize + tx];
    __syncthreads();

    for (int k = 0; k < kTileSize; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  P[row * size + col] = Pvalue;
}

template <typename T> T divUp(T a, T b) { return (a + b - 1) / b; }
} // namespace

void launchExercise1(float *P, const float *M, const float *N, size_t size) {
  dim3 const grid(divUp(size, kTileSize), divUp(size, kTileSize));
  dim3 const block(kTileSize, kTileSize);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  exercise1<<<grid, block, 0, stream>>>(P, M, N, size);
}