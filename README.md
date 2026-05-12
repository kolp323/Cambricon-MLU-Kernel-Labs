# Cambricon MLU Kernel Labs

Cambricon-MLU-Kernel-Labs contains two Cambricon MLU programming experiments: a PyTorch custom sigmoid operator built with MLUExtension, and a Bang C matrix-multiplication optimization sequence. The code covers both framework-level operator integration and lower-level accelerator kernel optimization.

## Technical Highlights

- **PyTorch MLU custom operator**: builds a custom sigmoid operator through `torch_mlu.utils.cpp_extension.MLUExtension` and exposes it to Python with PyBind11.
- **Bang C device kernel**: implements sigmoid with GDRAM-to-NRAM staging, BANG intrinsic activation and CNRT queue execution.
- **Autograd wrapper**: wraps the MLU operator in `torch.autograd.Function` and checks forward/backward behavior against PyTorch sigmoid.
- **MatMul optimization ladder**: compares scalar kernels, NRAM staging, `__bang_matmul`, block parallelism, pipeline scheduling, SRAM staging and UNION launches.
- **Hardware-oriented measurement**: records timing results from an MLU370-M8 environment for comparing optimization stages.

## Repository Layout

```text
Cambricon-MLU-Kernel-Labs/
|-- custom_pytorch_mlu_op/     # PyTorch + torch_mlu custom sigmoid operator
|-- matmul_optimization/       # Six Bang C matrix multiplication optimization variants
|-- docs/                      # Optimization notes and environment constraints
|-- .gitignore
`-- README.md
```

## Components

| Area | Path | Purpose |
| --- | --- | --- |
| Custom PyTorch MLU op | `custom_pytorch_mlu_op/` | Connects PyTorch tensors, C++ extension code, CNRT queues and a Bang C sigmoid kernel |
| MatMul kernel optimization | `matmul_optimization/` | Builds matrix multiplication from a scalar baseline to SRAM/UNION pipelined kernels |

## Custom PyTorch MLU Operator

The custom operator module follows this path:

1. `custom_pytorch_mlu_op/setup.py` uses `MLUExtension` to compile C++ and Bang C sources.
2. `mlu_custom_ext/mlu/src/bang_sigmoid.cpp` bridges PyTorch tensors to the MLU runtime and launches the kernel.
3. `mlu_custom_ext/mlu/src/bang_sigmoid_sample.mlu` implements the device-side sigmoid with NRAM buffering and `__bang_active_sigmoid`.
4. `mlu_custom_ext/mlu_functions/mlu_functions.py` wraps the operator in a Python autograd function.
5. `custom_pytorch_mlu_op/tests/` compares the MLU operator with PyTorch sigmoid behavior and runs a benchmark test.

Run the custom operator tests from the module directory:

```bash
cd custom_pytorch_mlu_op
python setup.py install
python -m unittest tests/test_sigmoid.py
python -m unittest tests/test_sigmoid_benchmark.py
```

## Matrix Multiplication Optimization

The matrix multiplication module contains six Bang C kernels that add optimization techniques step by step.

| Kernel | Matrix size | Main technique | Recorded MLU time |
| --- | --- | --- | --- |
| `01_scalar.mlu` | `128 x 256 x 256` | scalar device-side loops | `5280.594 ms` |
| `02_scalar_nram.mlu` | `128 x 256 x 256` | NRAM staging | `193.940 ms` |
| `03_vector_nram.mlu` | `128 x 256 x 256` | `__bang_matmul` with NRAM/WRAM | `0.061 ms` |
| `04_vector_nram_blocks.mlu` | `8192 x 512 x 512` | block-level parallelism | `0.453 ms` |
| `05_vector_nram_blocks_pipe3.mlu` | `8192 x 512 x 512` | three-stage NRAM pipeline | `0.444 ms` |
| `06_vector_sram_unions_pipe5.mlu` | `8192 x 512 x 512` | SRAM staging, UNION launch and five-stage pipeline | `0.373 ms` |

The timing records were measured on an MLU370-M8 at 1300 MHz with Neuware 3.15.11. They are useful for comparing the optimization stages under the same local hardware/software environment.

Run the full sequence from the matrix multiplication directory:

```bash
cd matmul_optimization
source env.sh
bash test.sh
```

`test.sh` rebuilds each kernel with `cncc --bang-arch=compute_30 -O3` and runs the generated binaries from `build/`.

## Environment

The experiments require a Cambricon MLU development environment:

- Neuware toolkit with `cncc`, CNRT headers and runtime libraries,
- PyTorch with `torch_mlu` for the custom operator module,
- an MLU370-class device for reproducing the recorded timings,
- standard Python build tooling for `setup.py`.

`matmul_optimization/env.sh` uses `NEUWARE_HOME` when it is already set, otherwise it defaults to `/torch/neuware_home`:

```bash
export NEUWARE_HOME=/path/to/neuware
source matmul_optimization/env.sh
```

Adjust `NEUWARE_HOME`, `PATH` and `LD_LIBRARY_PATH` for the local Neuware installation before building.

## Development Notes

- Generated build products are written under module-local `build/` directories.
- `custom_pytorch_mlu_op/tests/` checks the custom sigmoid forward and backward paths.
- `docs/optimization_notes.md` summarizes the relationship between framework integration, memory hierarchy and kernel scheduling.
