import torch
import numpy as np
import time
torch.ops.load_library("chapter6.so")
torch.manual_seed(0)

ITERS = 10000

def run_chap6(C, A, B):
    C.to(device="cuda")
    A.to(device="cuda")
    BT = B.T.contiguous().to(device="cuda")
    for _ in range(ITERS):
        torch.ops.chapter6.exercise1(C, A, BT)
    
def test_mm(func, shapeA, shapeB):
    A = torch.randn(shapeA, device="cuda")
    B = torch.randn(shapeB, device="cuda")
    C = torch.randn((shapeA[0], shapeB[1]), device="cuda")
    start = time.time()
    func(C, A, B)
    end = time.time()
    print(f"Time taken for {shapeA} x {shapeB} = {end-start:.6f} seconds")
    # delta = (C - torch.matmul(A, B)).cpu().detach().numpy()
    # print(delta)
    assert torch.allclose(C, torch.matmul(A, B), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    # Only square matrices are supported
    sizes = [
        1, 2, 3, 32, 64, 254, 255, 256, 1023, 1024, 1025
    ]
    for size in sizes:
        if size % 32 == 0:
            print(f"Testing chap6 at {size=}")
            test_mm(run_chap6, (size, size), (size, size))
        print(f"Passed all tests for size = {size}")    