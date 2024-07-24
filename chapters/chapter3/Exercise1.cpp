#include "Exercise1.hpp"
#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <random>
#include <vector>

#include "View.hpp"

namespace terrelln {
void matrixMultiplyCPU(View2D<float> const &P, View2D<float const> const &M,
                       View2D<float const> const &N) {
  auto const rows = P.shape(0);
  auto const cols = P.shape(1);
  auto const inner = M.shape(1);
  if (M.shapes() != std::array{rows, inner} ||
      N.shapes() != std::array{inner, cols}) {
    throw std::runtime_error("Matrix dimensions do not match");
  }
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      float sum = 0;
      for (size_t k = 0; k < inner; ++k) {
        sum += M.at(row, k) * N.at(k, col);
      }
      P.at(row, col) = sum;
    }
  }
}

template <typename Fn>
void __attribute__((noinline)) benchmark(std::string_view name, size_t reps,
                                         Fn &&fn) {
  auto start = std::chrono::high_resolution_clock::now();
  fn(reps);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto const durPerRep = duration.count() / reps;

  std::cout << std::format("{}: {}ms\n", name, durPerRep);
}

void fill(std::vector<float> &data) {
  std::mt19937 gen(0xdeadbeef ^ 0xcafebabe);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::generate(data.begin(), data.end(), [&] { return dist(gen); });
}

void validate(View2D<float const> P0, View2D<float const> P1) {
  if (P0.shapes() != P1.shapes()) {
    throw std::runtime_error("Invalid shapes!");
  }
  auto const [rows, cols] = P0.shapes();
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      auto const epsilon = 1e-4;
      auto v0 = P0.at(row, col);
      auto v1 = P1.at(row, col);
      if (std::abs(v0 - v1) > epsilon) {
        throw std::runtime_error(
            std::format("Mismatch at {}, {} ({} vs {})", row, col, v0, v1));
      }
    }
  }
}

void benchmark() {
  size_t constexpr M_rows = 1024;
  size_t constexpr M_cols = 1024;
  size_t constexpr N_rows = M_cols;
  size_t constexpr N_cols = 1024;
  size_t constexpr P_rows = M_rows;
  size_t constexpr P_cols = N_cols;

  std::vector<float> M_data(M_rows * M_cols);
  std::vector<float> N_data(N_rows * N_cols);
  std::vector<float> P0_data(P_rows * P_cols);
  std::vector<float> P1_data(P_rows * P_cols);
  std::vector<float> P2_data(P_rows * P_cols);

  fill(M_data);
  fill(N_data);

  auto M = view::rowMajor<float, M_rows, M_cols>(M_data.data(), M_rows, M_cols);
  auto N = view::rowMajor<float, N_rows, N_cols>(N_data.data(), N_rows, N_cols);
  auto P0 =
      view::rowMajor<float, P_rows, P_cols>(P0_data.data(), P_rows, P_cols);
  auto P1 =
      view::rowMajor<float, P_rows, P_cols>(P1_data.data(), P_rows, P_cols);
  auto P2 =
      view::rowMajor<float, P_rows, P_cols>(P2_data.data(), P_rows, P_cols);

  benchmark("CPU", 3, [&](size_t reps) {
    for (size_t i = 0; i < reps; ++i) {
      matrixMultiplyCPU(P0, M, N);
    }
  });

  benchmark("Row", 25, [&](size_t reps) {
    matrixMultiplyByRow(P1.data(), M.data(), N.data(), M.shape(0), M.shape(1),
                        N.shape(0), N.shape(1), reps);
  });
  benchmark("Cols", 25, [&](size_t reps) {
    matrixMultiplyByCol(P2.data(), M.data(), N.data(), M.shape(0), M.shape(1),
                        N.shape(0), N.shape(1), reps);
  });

  std::cout << std::format("Validating rows") << std::endl;
  validate(P0, P1);
  std::cout << std::format("Validating cols") << std::endl;
  validate(P0, P2);
}
} // namespace terrelln

int main(int argc, char **argv) { terrelln::benchmark(); }