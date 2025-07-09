#include "../utility/hip_utility.hpp"
#include <chrono>

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

void initialize_matrix(int *matrix, int M, int N);
void print_matrix(int *matrix, int M, int N);
void cpu_gemm(int *A, int *B, int *C, int M, int N, int K);
void compare_matrices(int *C_gpu, int *C_cpu, int M, int N);
int main() {

    // set the matrix size
    int M = 4096, N = 2048, K = 2048;

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
    // std::cout << "A:" << std::endl;
    // print_matrix(A_host, M, K);
    // std::cout << "B:" << std::endl;
    // print_matrix(B_host, K, N);
    // std::cout << "C:" << std::endl;
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

void initialize_matrix(int *matrix, int M, int N)
{
    printf("Initializing matrix...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = rand() % 10;
        }
    }
}
void print_matrix(int *matrix, int M, int N)
{
    printf("Printing matrix...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}
void cpu_gemm(int *A, int *B, int *C, int M, int N, int K)
{   
    printf("Running CPU GEMM...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
void compare_matrices(int *C_gpu, int *C_cpu, int M, int N)
{
    printf("Comparing matrices...\n");
    int error_count = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (C_gpu[i * N + j] != C_cpu[i * N + j]) {
                if (error_count < 10) {
                    printf("Error at position (%d, %d): GPU = %d, CPU = %d\n", i, j, C_gpu[i * N + j], C_cpu[i * N + j]);
                }
                error_count++;
            }
        }
    }
    if (error_count == 0) {
        printf("Matrices are equal\n");
    } else {
        printf("Matrices are not equal\n");
        printf("Error count: %d\n", error_count);
        printf("Error percentage: %.2f%%\n", (float)error_count / (M * N) * 100);
    }
}