#include "chapter6.h"
#include "ATen/ATen.h"
#include "torch/extension.h"

void exercise1(at::Tensor C, at::Tensor A, at::Tensor B) {
    launchExercise1(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), C.size(0));
}

TORCH_LIBRARY(chapter6, m) {
  m.def("exercise1", exercise1);
}