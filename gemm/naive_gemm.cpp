#include "../utility/hip_utility.hpp"
#include "../utility/ops_gemm.hpp"
#include <hip/hip_fp16.h>
#include <chrono>
#include <cstdlib>  // for atoi
#include <vector>
#include <cmath>    // for std::abs
#include <algorithm> // for std::max

using float16_t = _Float16;

__global__ void gemmKernel(float16_t *A, float16_t *B, float16_t *C, int M, int N, int K) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        float16_t sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char* argv[]) {

    // set the matrix size from command line arguments or use defaults
    int M = 1024, N = 1024, K = 1024;  // default values
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    } else if (argc == 2) {
        // If only one argument, use it for all dimensions (square matrices)
        M = N = K = atoi(argv[1]);
    } else if (argc > 1) {
        printf("Usage: %s [M N K] or %s [size]\n", argv[0], argv[0]);
        printf("  M N K: Matrix dimensions (A: M×K, B: K×N, C: M×N)\n");
        printf("  size:  Use same size for all dimensions (size×size)\n");
        printf("Using default size: %d×%d×%d\n", M, N, K);
    }
    
    printf("Matrix dimensions: A(%d×%d) × B(%d×%d) = C(%d×%d)\n", M, K, K, N, M, N);

    // allocate memory for the matrices
    float16_t *A_host = new float16_t[M * K];
    float16_t *B_host = new float16_t[K * N];
    float16_t *C_host = new float16_t[M * N];
    float16_t *C_host_cpu = new float16_t[M * N];

    // initialize the matrices
    fillRand(A_host, M, K);
    fillRand(B_host, K, N);
    fillRand(C_host, M, N);

    // print A, B, C
    // print_matrix(A_host, M, K);
    // print_matrix(B_host, K, N);
    // print_matrix(C_host, M, N);

    // allocate memory for the matrices on the device
    float16_t *A_device, *B_device, *C_device;
    HIP_CHECK(hipMalloc(&A_device, M * K * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&B_device, K * N * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&C_device, M * N * sizeof(float16_t)));

    // copy the matrices from the host to the device
    HIP_CHECK(hipMemcpy(A_device, A_host, M * K * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_device, B_host, K * N * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(C_device, C_host, M * N * sizeof(float16_t), hipMemcpyHostToDevice));

    // set the block and grid size
    dim3 block(16, 16);  // 256 threads per block - much better for GPU utilization
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    // print number of blocks and threads
    printf("grid: %d, %d\n", grid.x, grid.y);
    printf("block: %d, %d\n", block.x, block.y);
    printf("number of blocks: %d\n", grid.x * grid.y);
    printf("number of threads: %d\n", block.x * block.y);

    // Create HIP events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Record start event
    HIP_CHECK(hipEventRecord(start, 0));
    
    // launch the kernel
    gemmKernel<<<grid, block>>>(A_device, B_device, C_device, M, N, K);
    
    // Record stop event
    HIP_CHECK(hipEventRecord(stop, 0));
    
    // CPU GEMM for comparison (using float for CPU reference)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> A_float(M * K);
    std::vector<float> B_float(K * N);
    std::vector<float> C_float(M * N, 0.0f);
    std::vector<float> D_float(M * N);
    
    // Convert FP16 to FP32 for CPU computation
    for (int i = 0; i < M * K; i++) A_float[i] = static_cast<float>(A_host[i]);
    for (int i = 0; i < K * N; i++) B_float[i] = static_cast<float>(B_host[i]);
    
    // Use template GEMM: D = 1.0 * A*B + 0.0 * C
    gemm_cpu_h<float, float, float, row_major, row_major, row_major>(
        M, N, K,
        A_float.data(), B_float.data(), C_float.data(), D_float.data(),
        K, N, N, N,  // lda, ldb, ldc, ldd
        1.0f, 0.0f   // alpha, beta
    );
    
    // Convert back to FP16
    for (int i = 0; i < M * N; i++) C_host_cpu[i] = static_cast<float16_t>(D_float[i]);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    // wait for the kernel to complete
    HIP_CHECK(hipDeviceSynchronize());
    
    // Calculate GPU time
    float gpu_time_ms;
    HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, start, stop));
    
    // Calculate CPU time
    auto cpu_time_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // Calculate performance metrics
    double gflops = (2.0 * M * N * K) / (gpu_time_ms * 1e6);  // GFLOPS
    double tflops = gflops / 1000.0;  // TFLOPS
    
    // Calculate memory bandwidth (GB/s)
    // Reads: A (M*K), B (K*N)  |  Writes: C (M*N)
    size_t bytes_transferred = (M * K + K * N + M * N) * sizeof(float16_t);
    double bandwidth_gbs = (bytes_transferred / 1e9) / (gpu_time_ms / 1000.0);
    
    printf("\n=== Performance Results ===\n");
    printf("Matrix: A(%d×%d) × B(%d×%d) = C(%d×%d)\n", M, K, K, N, M, N);
    printf("GPU Time: %.3f ms\n", gpu_time_ms);
    printf("GPU Performance: %.2f GFLOPS (%.4f TFLOPS)\n", gflops, tflops);
    printf("Memory Bandwidth: %.2f GB/s\n", bandwidth_gbs);
    printf("CPU Time: %.3f ms\n", cpu_time_ms);
    printf("Speedup: %.2fx\n", cpu_time_ms / gpu_time_ms);
    
    // copy the result back from the device to the host
    HIP_CHECK(hipMemcpy(C_host, C_device, M * N * sizeof(float16_t), hipMemcpyDeviceToHost));
    
    // Compare results (convert to float for comparison)
    std::vector<float> C_gpu_float(M * N);
    std::vector<float> C_cpu_float(M * N);
    for (int i = 0; i < M * N; i++) {
        C_gpu_float[i] = static_cast<float>(C_host[i]);
        C_cpu_float[i] = static_cast<float>(C_host_cpu[i]);
    }
    
    // Custom validation for FP16-originated data
    // FP16 epsilon ≈ 0.001, allowing 50× gives ~5% tolerance (reasonable for K=1024 accumulations)
    double max_error = 0.0;
    bool passed = true;
    const double fp16_epsilon = 0.0009765625; // 2^-10, actual FP16 epsilon
    const double threshold = 50.0 * fp16_epsilon; // ~0.05 = 5%
    
    for (int i = 0; i < M * N; i++) {
        double diff = std::abs(C_gpu_float[i] - C_cpu_float[i]);
        double scale = std::abs(C_gpu_float[i]) + std::abs(C_cpu_float[i]) + 1.0;
        double rel_error = diff / scale;
        max_error = std::max(max_error, rel_error);
        if (rel_error > threshold) {
            passed = false;
        }
    }
    
    if (passed) {
        printf("PASSED! Max relative error: %e (threshold: %e)\n", max_error, threshold);
    } else {
        printf("FAILED! Max relative error: %e (threshold: %e)\n", max_error, threshold);
    }
    
    // free the memory on the device
    HIP_CHECK(hipFree(A_device));
    HIP_CHECK(hipFree(B_device));
    HIP_CHECK(hipFree(C_device));
    
    // Clean up events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // free the memory on the host
    delete[] A_host;
    delete[] B_host;
    delete[] C_host;

    return 0;
}

