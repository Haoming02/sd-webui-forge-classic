## Forge Neo GGUF Kernels

This directory contains Forge's PyTorch quantization and dequantization kernels based on `llama.cpp`. Forge uses the stock `gguf` package for file parsing and quantization constants.

Keep this directory as a namespace package so `import gguf` resolves to the installed package when Forge adds `modules_forge/packages` to `sys.path`.
