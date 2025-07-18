# Option Pricing GPU

A high-performance, CUDA-accelerated Monte Carlo simulator for pricing European, Asian, and Barrier options using GPU parallelism. This project uses a C++/CUDA backend with Python bindings (via PyTorch extension) to enable fast option pricing workflows directly from Python.

## Features

- European Call and Put Option Pricing  
- Asian Option Pricing (Arithmetic Average)  
- Barrier Option Pricing (Up-and-Out, Down-and-Out)  
- GPU acceleration using CUDA  
- Easily callable from Python  
- Compatible with Google Colab (for testing)

---

## Installation

### Requirements

- Python 3.8+
- CUDA Toolkit (>=11.0)
- PyTorch (with CUDA support)
- A GPU with CUDA support

### Build Extension

Build the PyTorch C++/CUDA extension in-place:

```bash
python setup.py build_ext --inplace
```
