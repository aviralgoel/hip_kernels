# GPU RMS Normalization

Practice implementation for [LeetGPU RMS Normalization](https://leetgpu.com/challenges/rms-normalization).

## Building

From the project root:

```bash
cd /home/AMD/avirgoel/personal/leetgpu/hip_kernels
mkdir -p build && cd build
cmake ..
make 01_rms_normalization
```

## Running

```bash
cd build/rms
./01_rms_normalization          # default N=1,000,000
./01_rms_normalization 4        # small case for debugging
./01_rms_normalization 100000   # LeetGPU performance size
```

Each run generates random input, compares GPU output against `cpu_rms_norm` from `utility/cpu_verify.hpp`, and reports timing plus max error.

## Shared utilities

| Header | Purpose |
|--------|---------|
| `utility/cpu_verify.hpp` | CPU reference implementations (`cpu_rms_norm`) |
| `utility/vector_match.hpp` | Vector comparison helpers (`max_vector_error`, `vectors_match`) |
