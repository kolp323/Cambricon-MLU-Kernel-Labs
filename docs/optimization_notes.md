# Notes

## Why These Experiments Are Grouped Together

Both modules exercise the same MLU programming stack from different levels:

- the custom operator path connects PyTorch tensors, C++ extension code, CNRT queues and Bang C kernels;
- the matrix multiplication path focuses on kernel-level memory hierarchy and execution scheduling.

Together they show the movement from framework integration to lower-level accelerator optimization.

## Optimization Concepts

The matrix multiplication sequence highlights several MLU-specific concerns:

- moving data from GDRAM into NRAM or WRAM before computation;
- replacing scalar loops with `__bang_matmul`;
- splitting work across blocks and UNION launches;
- using asynchronous memory copies and staged pipelines;
- using SRAM as a shared staging layer for larger matrix tiles.

## Reproducibility Boundaries

The code depends on Cambricon Neuware, `cncc`, CNRT and `torch_mlu`, so it is not expected to build on a normal CPU-only machine. Recorded timings should be read as local experiment evidence for the listed hardware/software environment.
