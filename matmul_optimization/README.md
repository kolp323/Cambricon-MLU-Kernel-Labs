# Matrix Multiplication Optimization

This directory contains six Bang C matrix multiplication kernels that build up optimization techniques step by step.

## Kernel Stages

| File | Technique |
| --- | --- |
| `01_scalar.mlu` | baseline scalar matrix multiplication in an MLU kernel |
| `02_scalar_nram.mlu` | scalar computation after staging matrices in NRAM |
| `03_vector_nram.mlu` | vectorized `__bang_matmul` with NRAM and WRAM buffers |
| `04_vector_nram_blocks.mlu` | block-level parallel execution across matrix rows |
| `05_vector_nram_blocks_pipe3.mlu` | asynchronous NRAM pipeline over block chunks |
| `06_vector_sram_unions_pipe5.mlu` | SRAM staging, UNION launch and deeper pipeline scheduling |

## Recorded Timing

The preserved run was measured on MLU370-M8 at 1300 MHz with Neuware 3.15.11.

| Kernel | MLU time |
| --- | --- |
| `01_scalar.mlu` | `5280.594 ms` |
| `02_scalar_nram.mlu` | `193.940 ms` |
| `03_vector_nram.mlu` | `0.061 ms` |
| `04_vector_nram_blocks.mlu` | `0.453 ms` |
| `05_vector_nram_blocks_pipe3.mlu` | `0.444 ms` |
| `06_vector_sram_unions_pipe5.mlu` | `0.373 ms` |

## Run

```bash
source env.sh
bash test.sh
```

Generated binaries are written to `build/` and are intentionally ignored by Git.
