#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "../utility/cpu_verify.hpp"
#include "../utility/hip_utility.hpp"
#include "../utility/vector_match.hpp"

#define THREADS_PER_BLOCK 256

__global__ void myRMSKernel(const float* input, float gamma, float beta, float epsilon, int N,
                            float* output) {
    int elementsPerThread = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    __shared__ float shared_sum[THREADS_PER_BLOCK];
    shared_sum[threadIdx.x] = 0;
    __syncthreads();

    for (int i = 0; i < elementsPerThread; i++) {
        if (threadIdx.x + (blockDim.x * i) < N) {
            float temp =
                (input[threadIdx.x + (blockDim.x * i)] * input[threadIdx.x + (blockDim.x * i)]);
            shared_sum[threadIdx.x] += temp;
        }
    }
    __syncthreads();

    for (int x = THREADS_PER_BLOCK / 2; x > 0; x /= 2) {
        if (threadIdx.x < x) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + x];
        }
        __syncthreads();
    }

    float rms_0 = shared_sum[0];
    float rms_1 = rms_0 / N;
    float rms_2 = sqrtf(rms_1 + epsilon);

    for (int i = 0; i < elementsPerThread; i++) {
        if (blockDim.x * i + threadIdx.x < N) {
            float xi_cap = input[threadIdx.x + (blockDim.x * i)] / rms_2;
            output[threadIdx.x + (blockDim.x * i)] = gamma * xi_cap + beta;
        }
    }
}

void solve(const float* input, float gamma, float beta, float* output, int N, float eps) {
    hipLaunchKernelGGL(myRMSKernel, dim3(1), dim3(THREADS_PER_BLOCK), 0, 0, input, gamma,
                       beta, eps, N, output);
    HIP_CHECK(hipDeviceSynchronize());
}

int main(int argc, char* argv[]) {
    int N = 1000000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    const float gamma = 1.0f;
    const float beta = 0.0f;
    const float eps = 1e-5f;
    const float tol = 1e-4f;

    float* h_input = new float[N];
    float* h_output_gpu = new float[N];
    float* h_output_cpu = new float[N];

    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_rms_norm(h_input, gamma, beta, eps, h_output_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, 0));
    solve(d_input, gamma, beta, d_output, N, eps);
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float gpu_time_ms;
    HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, start, stop));

    HIP_CHECK(hipMemcpy(h_output_gpu, d_output, N * sizeof(float), hipMemcpyDeviceToHost));

    const double max_error = max_vector_error(h_output_gpu, h_output_cpu, N);
    const bool passed = vectors_match(h_output_gpu, h_output_cpu, N, tol);

    printf("N=%d, CPU=%.3fms, GPU=%.3fms, Speedup=%.2fx, MaxError=%.2e, %s\n", N, cpu_time_ms,
           gpu_time_ms, cpu_time_ms / gpu_time_ms, max_error, passed ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;

    return passed ? 0 : 1;
}
