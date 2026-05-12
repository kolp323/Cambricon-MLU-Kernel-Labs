# Cambricon MLU Kernel Labs

This repository collects two Cambricon MLU programming experiments: a PyTorch custom operator implemented with MLUExtension, and a progressive matrix-multiplication optimization sequence written in Bang C.

The code was extracted from an AI computing systems coursework repository and reorganized as a focused public engineering project. Generated build directories, package metadata, course archives and duplicate submission folders are excluded.

## Technical Highlights

- **PyTorch MLU custom operator**: packages a custom sigmoid operator through `torch_mlu.utils.cpp_extension.MLUExtension` and exposes it to Python with PyBind11.
- **Bang C kernel implementation**: implements the sigmoid device kernel with GDRAM/NRAM transfers, BANG intrinsic activation and CNRT queue execution.
- **Autograd integration**: wraps the custom MLU sigmoid in `torch.autograd.Function`, with a Python backward path based on the sigmoid derivative.
- **MatMul optimization ladder**: compares scalar execution, NRAM staging, vectorized `__bang_matmul`, block parallelism, pipeline staging and SRAM/UNION optimization.
- **Hardware-oriented benchmarking**: keeps recorded timing results for MLU370-M8 and Hygon C86 CPU comparison.

## Repository Layout

```text
Cambricon-MLU-Kernel-Labs/
|-- custom_pytorch_mlu_op/     # PyTorch + torch_mlu custom sigmoid operator
|-- matmul_optimization/       # Six Bang C matrix multiplication optimization variants
|-- docs/                      # Notes on optimization stages and environment
|-- .gitignore
`-- README.md
```

## Components

| Area | Path | What it demonstrates |
| --- | --- | --- |
| Custom PyTorch MLU op | `custom_pytorch_mlu_op/` | MLUExtension packaging, C++ bridge, Bang C kernel entry and Python autograd wrapper |
| MatMul kernel optimization | `matmul_optimization/` | Step-by-step MLU matrix multiplication optimization from scalar kernel to SRAM/UNION pipelining |

## MatMul Optimization Summary

The preserved run log was measured on an MLU370-M8 at 1300 MHz with Neuware 3.15.11.

| Kernel | Matrix size | Main technique | MLU time |
| --- | --- | --- | --- |
| `01_scalar.mlu` | `128 x 256 x 256` | scalar device-side loops | `5280.594 ms` |
| `02_scalar_nram.mlu` | `128 x 256 x 256` | NRAM staging | `193.940 ms` |
| `03_vector_nram.mlu` | `128 x 256 x 256` | `__bang_matmul` with NRAM/WRAM | `0.061 ms` |
| `04_vector_nram_blocks.mlu` | `8192 x 512 x 512` | block parallelism | `0.453 ms` |
| `05_vector_nram_blocks_pipe3.mlu` | `8192 x 512 x 512` | three-stage NRAM pipeline | `0.444 ms` |
| `06_vector_sram_unions_pipe5.mlu` | `8192 x 512 x 512` | SRAM staging, UNION launch and five-stage pipeline | `0.373 ms` |

These values are experiment records from a specific local hardware/software environment, not portable benchmark claims.

## Environment

The code targets a Cambricon MLU environment with:

- Neuware toolkit and `cncc`,
- PyTorch with `torch_mlu`,
- CNRT runtime headers/libraries,
- an MLU370-class device for the recorded results.

The original recorded environment used Neuware `3.15.11` and MLU370-M8. Paths in `env.sh` may need adjustment for a different installation.

## Run

Build and test the custom PyTorch MLU operator:

```bash
cd custom_pytorch_mlu_op
python setup.py install
python -m unittest tests/test_sigmoid.py
python -m unittest tests/test_sigmoid_benchmark.py
```

Run the matrix multiplication optimization sequence:

```bash
cd matmul_optimization
source env.sh
bash test.sh
```

## Public Release Boundaries

This repository keeps source code, tests, build scripts and representative timing records. It excludes generated build outputs, package metadata, course archives and duplicate upload folders.
