#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

#define HIP_CHECK(expression)                                                  \
  {                                                                            \
    const hipError_t status = expression;                                      \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      exit(-1);                                                                \
    }                                                                          \
  }

__global__ void gemmKernel(int *A, int *B, int *C, int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // PRINT BLOCK ID AND THREAD ID AND ROW AND COL
    printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d, row: %d, col: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col);


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
int main() {

    // set the matrix size
    int M = 5, N = 5, K = 5;

    // allocate memory for the matrices
    int *A_host = new int[M * K];
    int *B_host = new int[K * N];
    int *C_host = new int[M * N];

    // initialize the matrices
    initialize_matrix(A_host, M, K);
    initialize_matrix(B_host, K, N);
    initialize_matrix(C_host, M, N);

    // print A, B, C
    std::cout << "A:" << std::endl;
    print_matrix(A_host, M, K);
    std::cout << "B:" << std::endl;
    print_matrix(B_host, K, N);
    std::cout << "C:" << std::endl;
    print_matrix(C_host, M, N);

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
    dim3 block(3, 3);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    // print number of blocks and threads
    printf("grid: %d, %d\n", grid.x, grid.y);
    printf("block: %d, %d\n", block.x, block.y);
    printf("number of blocks: %d\n", grid.x * grid.y);
    printf("number of threads: %d\n", block.x * block.y);

    // launch the kernel
    gemmKernel<<<grid, block>>>(A_device, B_device, C_device, M, N, K);
    
    // wait for the kernel to complete
    HIP_CHECK(hipDeviceSynchronize());
    
    // copy the result back from the device to the host
    HIP_CHECK(hipMemcpy(C_host, C_device, M * N * sizeof(int), hipMemcpyDeviceToHost));
    
    // free the memory on the device
    HIP_CHECK(hipFree(A_device));
    HIP_CHECK(hipFree(B_device));
    HIP_CHECK(hipFree(C_device));

    // free the memory on the host
    delete[] A_host;
    delete[] B_host;
    delete[] C_host;

    return 0;
}

void initialize_matrix(int *matrix, int M, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = rand() % 10;
        }
    }
}
void print_matrix(int *matrix, int M, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}