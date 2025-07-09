#include "../utility/hip_utility.hpp"
#include "../utility/ops_gemm.hpp"
#include <chrono>
#include <cstdlib>  // for atoi

__global__ void gemmKernel(int *A, int *B, int *C, int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int sum = 0;
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
    int *A_host = new int[M * K];
    int *B_host = new int[K * N];
    int *C_host = new int[M * N];
    int *C_host_cpu = new int[M * N];

    // initialize the matrices
    initialize_matrix(A_host, M, K);
    initialize_matrix(B_host, K, N);
    initialize_matrix(C_host, M, N);

    // print A, B, C
    // print_matrix(A_host, M, K);
    // print_matrix(B_host, K, N);
    // print_matrix(C_host, M, N);

    // allocate memory for the matrices on the device
    int *A_device, *B_device, *C_device;
    HIP_CHECK(hipMalloc(&A_device, M * K * sizeof(int)));
    HIP_CHECK(hipMalloc(&B_device, K * N * sizeof(int)));
    HIP_CHECK(hipMalloc(&C_device, M * N * sizeof(int)));

    // copy the matrices from the host to the device
    HIP_CHECK(hipMemcpy(A_device, A_host, M * K * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_device, B_host, K * N * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(C_device, C_host, M * N * sizeof(int), hipMemcpyHostToDevice));

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
    
    // CPU GEMM for comparison
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemm(A_host, B_host, C_host_cpu, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    // print_matrix(C_host_cpu, M, N);
    
    // wait for the kernel to complete
    HIP_CHECK(hipDeviceSynchronize());
    
    // Calculate GPU time
    float gpu_time_ms;
    HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, start, stop));
    
    // Calculate CPU time
    auto cpu_time_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    printf("GPU Time: %.3f ms\n", gpu_time_ms);
    printf("CPU Time: %.3f ms\n", cpu_time_ms);
    printf("Speedup: %.2fx\n", cpu_time_ms / gpu_time_ms);
    
    // copy the result back from the device to the host
    HIP_CHECK(hipMemcpy(C_host, C_device, M * N * sizeof(int), hipMemcpyDeviceToHost));
    compare_matrices(C_host, C_host_cpu, M, N);
    
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

