# GPU Softmax Implementation

This directory contains implementations of the softmax function on AMD GPUs using HIP.

## Current Implementation

### 01: Naive Softmax (Your Implementation)
**The approach:** Single-block reduction for max and sum, then normalize.

## Building

From the project root:

```bash
cd /home/avirgoel/aviral/hip_kernels
mkdir -p build && cd build
cmake ..
make
```

## Running

```bash
cd build/softmax
./01_naive_softmax [N]
```

Examples:
```bash
# Small size for debugging (prints input/output)
./01_naive_softmax 5

# Medium size
./01_naive_softmax 1000

# Large size
./01_naive_softmax 500000
```