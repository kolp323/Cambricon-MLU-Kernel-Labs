# Custom PyTorch MLU Operator

This module builds a custom sigmoid operator for Cambricon MLU and exposes it as a Python function.

## Implementation Path

1. `setup.py` uses `torch_mlu.utils.cpp_extension.MLUExtension` to compile C++ and Bang C sources.
2. `mlu_custom_ext/mlu/src/bang_sigmoid.cpp` bridges PyTorch tensors to the MLU runtime and launches the kernel entry.
3. `mlu_custom_ext/mlu/src/bang_sigmoid_sample.mlu` implements the device kernel with NRAM buffering and `__bang_active_sigmoid`.
4. `mlu_custom_ext/mlu_functions/mlu_functions.py` wraps the operator in `torch.autograd.Function`.
5. `tests/` compares forward and backward behavior with PyTorch CPU sigmoid.

## Run

```bash
python setup.py install
python -m unittest tests/test_sigmoid.py
python -m unittest tests/test_sigmoid_benchmark.py
```

A working Neuware, PyTorch and `torch_mlu` environment is required.
