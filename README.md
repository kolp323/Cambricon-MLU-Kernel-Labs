# Cambricon MLU Kernel Labs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*(English version follows the Chinese version)*

---

## 🇨🇳 中文版

`Cambricon-MLU-Kernel-Labs` 包含两个寒武纪 MLU (Machine Learning Unit) 编程实验项目：一个是基于 `MLUExtension` 构建的 PyTorch 自定义 Sigmoid 算子，另一个是基于 Bang C 的矩阵乘法 (MatMul) 渐进式优化实践。本项目代码全面覆盖了从上层深度学习框架的算子集成，到底层 AI 加速器的硬件级内核优化。

### 🌟 技术亮点

- **PyTorch MLU 自定义算子**: 通过 `torch_mlu.utils.cpp_extension.MLUExtension` 构建自定义 Sigmoid 算子，并使用 PyBind11 将其暴露给 Python 层。
- **Bang C 设备端内核**: 实现了完整的 Sigmoid 运算流，包括 GDRAM 到 NRAM 的数据搬运、基于 BANG intrinsic 的激活函数调用以及 CNRT 队列执行。
- **自动求导 (Autograd) 封装**: 在 `torch.autograd.Function` 中封装 MLU 算子，并与原生 PyTorch 的 Sigmoid 进行前向和反向传播的精度对齐验证。
- **矩阵乘法进阶优化**: 包含从基础标量内核起步的 6 阶进阶优化方案，涵盖 NRAM 暂存、`__bang_matmul` 指令、块级并行 (Block parallelism)、流水线调度 (Pipeline scheduling)、SRAM 暂存以及 UNION 任务启动。
- **面向硬件的性能评估**: 在基于 Neuware 3.15.11 的 MLU370-M8 (频率 1300 MHz) 环境中详细记录了各优化阶段的耗时，直观展示了硬件优化的性能收益。

### 📁 目录结构

```text
Cambricon-MLU-Kernel-Labs/
|-- custom_pytorch_mlu_op/     # PyTorch + torch_mlu 自定义 Sigmoid 算子
|-- matmul_optimization/       # Bang C 矩阵乘法 6 阶性能优化代码
|-- docs/                      # 优化笔记及环境配置说明
|-- LICENSE                    # 开源协议文件
`-- README.md                  # 本文档
```

#### 组件说明

| 模块 | 路径 | 功能说明 |
| --- | --- | --- |
| **自定义 PyTorch MLU 算子** | `custom_pytorch_mlu_op/` | 将 PyTorch Tensor、C++ 扩展层、CNRT 任务队列以及 Bang C 内核打通 |
| **矩阵乘法内核优化** | `matmul_optimization/` | 实现从原生标量计算到基于 SRAM/UNION 流水线的高效矩阵乘法内核 |

### 🛠️ 环境要求

实验需要配置标准的寒武纪 MLU 开发环境：
- 包含 `cncc` 编译器、CNRT 头文件及运行时库的 **Neuware 工具包**。
- 支持 `torch_mlu` 的 PyTorch 版本（用于自定义算子模块）。
- MLU370 系列加速卡（用于复现并进行基准性能测试）。
- 标准的 Python 编译工具（用于运行 `setup.py`）。

编译前请确保正确配置环境变量。如果系统中未设置 `NEUWARE_HOME`，默认路径将使用 `/torch/neuware_home`。请根据实际情况调整 `NEUWARE_HOME`、`PATH` 和 `LD_LIBRARY_PATH`：

```bash
export NEUWARE_HOME=/path/to/your/neuware
source matmul_optimization/env.sh
```

### 🚀 快速开始

#### 1. 运行 PyTorch MLU 自定义算子

该模块执行流程为：通过 `setup.py` 编译 C++ 及 Bang C 源码；在 `bang_sigmoid.cpp` 中打通 Tensor 与 MLU runtime；通过 `bang_sigmoid_sample.mlu` 在设备端使用 NRAM 和 `__bang_active_sigmoid` 执行计算；最后在 Python 层封装 Autograd 功能。

请在自定义算子目录下运行测试和基准评估：

```bash
cd custom_pytorch_mlu_op
python setup.py install
python -m unittest tests/test_sigmoid.py
python -m unittest tests/test_sigmoid_benchmark.py
```

#### 2. 矩阵乘法进阶优化

矩阵乘法模块包含了 6 个逐步叠加优化策略的 Bang C 内核代码。以下为在 MLU370-M8 (1300 MHz) 上的测试数据参考：

| 内核代码 | 矩阵维度 | 核心优化技术 | 实测耗时 (MLU Time) |
| --- | --- | --- | --- |
| `01_scalar.mlu` | `128 x 256 x 256` | 标量设备端循环 | `5280.594 ms` |
| `02_scalar_nram.mlu` | `128 x 256 x 256` | NRAM 暂存 (Staging) | `193.940 ms` |
| `03_vector_nram.mlu` | `128 x 256 x 256` | `__bang_matmul` 与 NRAM/WRAM 结合 | `0.061 ms` |
| `04_vector_nram_blocks.mlu` | `8192 x 512 x 512` | 块级并行计算 (Block-level parallelism) | `0.453 ms` |
| `05_vector_nram_blocks_pipe3.mlu` | `8192 x 512 x 512` | 三级 NRAM 流水线并行 | `0.444 ms` |
| `06_vector_sram_unions_pipe5.mlu` | `8192 x 512 x 512` | SRAM 暂存、UNION 任务及五级流水线 | `0.373 ms` |

运行完整的矩阵乘法优化测试脚本：

```bash
cd matmul_optimization
source env.sh
bash test.sh
```
*注：`test.sh` 会自动使用 `cncc --bang-arch=compute_30 -O3` 编译内核，并运行 `build/` 目录下的可执行文件。*

### 📝 开发说明

- 编译产物统一存放在各子模块的 `build/` 目录下。
- 测试代码 `custom_pytorch_mlu_op/tests/` 提供了自定义 Sigmoid 算子的前向和反向传播正确性校验。
- 更多关于框架集成、内存分级与内核调度的深入解析，请参考 `docs/optimization_notes.md`。

### 📄 开源协议

本项目基于 [MIT License](LICENSE) 协议开源。

---

## 🇬🇧 English Version

`Cambricon-MLU-Kernel-Labs` contains two Cambricon MLU programming experiments: a PyTorch custom sigmoid operator built with `MLUExtension`, and a Bang C matrix-multiplication (MatMul) optimization sequence. The code covers both framework-level operator integration and lower-level accelerator kernel optimization.

### 🌟 Technical Highlights

- **PyTorch MLU Custom Operator**: Builds a custom sigmoid operator through `torch_mlu.utils.cpp_extension.MLUExtension` and exposes it to Python with PyBind11.
- **Bang C Device Kernel**: Implements sigmoid with GDRAM-to-NRAM staging, BANG intrinsic activation, and CNRT queue execution.
- **Autograd Wrapper**: Wraps the MLU operator in `torch.autograd.Function` and checks forward/backward behavior against PyTorch sigmoid.
- **MatMul Optimization Ladder**: Compares scalar kernels, NRAM staging, `__bang_matmul`, block parallelism, pipeline scheduling, SRAM staging, and UNION launches.
- **Hardware-oriented Measurement**: Records timing results from an MLU370-M8 environment for comparing optimization stages.

### 📁 Repository Layout

```text
Cambricon-MLU-Kernel-Labs/
|-- custom_pytorch_mlu_op/     # PyTorch + torch_mlu custom sigmoid operator
|-- matmul_optimization/       # Six Bang C matrix multiplication optimization variants
|-- docs/                      # Optimization notes and environment constraints
|-- LICENSE                    # License file
`-- README.md                  # This documentation
```

#### Components

| Area | Path | Purpose |
| --- | --- | --- |
| **Custom PyTorch MLU op** | `custom_pytorch_mlu_op/` | Connects PyTorch tensors, C++ extension code, CNRT queues and a Bang C sigmoid kernel |
| **MatMul kernel optimization**| `matmul_optimization/` | Builds matrix multiplication from a scalar baseline to SRAM/UNION pipelined kernels |

### 🛠️ Environment Prerequisites

The experiments require a Cambricon MLU development environment:
- Neuware toolkit with `cncc`, CNRT headers and runtime libraries.
- PyTorch with `torch_mlu` for the custom operator module.
- An MLU370-class device for reproducing the recorded timings.
- Standard Python build tooling for `setup.py`.

Before building, adjust `NEUWARE_HOME`, `PATH`, and `LD_LIBRARY_PATH` for the local Neuware installation. `matmul_optimization/env.sh` uses `NEUWARE_HOME` when it is already set; otherwise, it defaults to `/torch/neuware_home`:

```bash
export NEUWARE_HOME=/path/to/your/neuware
source matmul_optimization/env.sh
```

### 🚀 Getting Started

#### 1. Custom PyTorch MLU Operator

The custom operator module follows this execution path: `setup.py` compiles the source code, `bang_sigmoid.cpp` bridges tensors to the MLU runtime, `bang_sigmoid_sample.mlu` executes the device-side kernel, and Autograd handles Python-level gradients.

Run the custom operator tests from the module directory:

```bash
cd custom_pytorch_mlu_op
python setup.py install
python -m unittest tests/test_sigmoid.py
python -m unittest tests/test_sigmoid_benchmark.py
```

#### 2. Matrix Multiplication Optimization

The matrix multiplication module contains six Bang C kernels that add optimization techniques step by step. Below are the recorded MLU times measured on an MLU370-M8 at 1300 MHz with Neuware 3.15.11:

| Kernel | Matrix Size | Main Technique | Recorded MLU Time |
| --- | --- | --- | --- |
| `01_scalar.mlu` | `128 x 256 x 256` | Scalar device-side loops | `5280.594 ms` |
| `02_scalar_nram.mlu` | `128 x 256 x 256` | NRAM staging | `193.940 ms` |
| `03_vector_nram.mlu` | `128 x 256 x 256` | `__bang_matmul` with NRAM/WRAM | `0.061 ms` |
| `04_vector_nram_blocks.mlu` | `8192 x 512 x 512` | Block-level parallelism | `0.453 ms` |
| `05_vector_nram_blocks_pipe3.mlu` | `8192 x 512 x 512` | Three-stage NRAM pipeline | `0.444 ms` |
| `06_vector_sram_unions_pipe5.mlu` | `8192 x 512 x 512` | SRAM staging, UNION launch and five-stage pipeline | `0.373 ms` |

Run the full sequence from the matrix multiplication directory:

```bash
cd matmul_optimization
source env.sh
bash test.sh
```
*Note: `test.sh` rebuilds each kernel with `cncc --bang-arch=compute_30 -O3` and runs the generated binaries from `build/`.*

### 📝 Development Notes

- Generated build products are written under module-local `build/` directories.
- `custom_pytorch_mlu_op/tests/` checks the custom sigmoid forward and backward paths.
- `docs/optimization_notes.md` summarizes the relationship between framework integration, memory hierarchy, and kernel scheduling.

### 📄 License

This project is open-sourced under the [MIT License](LICENSE).
