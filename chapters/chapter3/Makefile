CXXFLAGS ?= -g -O3 -std=c++20
CPPFLAGS ?= -DNDEBUG -I/opt/cuda/include
LDFLAGS ?= -L/opt/cuda/lib 
CXX = clang++

NVCC = /opt/cuda/bin/nvcc
CUFLAGS ?= -ccbin=$(CXX)

# --cuda-path=/opt/cuda/ --cuda-gpu-arch=sm_80

SRCS := $(wildcard *.cu *.cpp)
HEADERS := $(wildcard *.hpp *.cuh)

Exercise1: Exercise1.cu Exercise1.cpp Exercise1.hpp
	$(NVCC) $(CUFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) Exercise1.cpp Exercise1.cu -o $@