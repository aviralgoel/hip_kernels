#include <hip/hip_runtime.h>
#include "../include/HIP/HIP_Utils.hpp"

#include <iostream>

#define TILE_SIZE 8

void generateMatrix(int *matrix, int rows, int cols);

__global__ void matrixMultiplication(int *A, int *B, int* C, int WIDTH)
{
    // initialize shared memory
    __shared__ int Mds[TILE_SIZE][TILE_SIZE];
    __shared__ int Nds[TILE_SIZE][TILE_SIZE];
    
    // thread Idx and thread Idy
    int tx = threadIdx.x; int ty = threadIdx.y;
    // block Idx and block Idy
    int bx = blockIdx.x; int by = blockIdx.y;

    // Row and Column
    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    int Pvalue = 0;
    for(int ph = 0; ph < WIDTH / TILE_SIZE; ph++)
    {
        // load data from global memory to shared memory
        Mds[ty][tx] = A[Row * WIDTH + ph * TILE_SIZE + tx];
        Nds[ty][tx] =  B[(ph * TILE_SIZE + ty) * WIDTH + Col];

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    C[Row * WIDTH + Col] = Pvalue;


}
int main()
{
    int M, N, K, WIDTH;
    M = N = K = WIDTH = 512;
    std::cout << "Matrix A: " << M << " x " << K << std::endl;
    std::cout << "Matrix B: " << K << " x " << N << std::endl;
    std::cout << "Matrix C: " << M << " x " << N << std::endl;

    int *h_A, *h_B, *h_C;

    h_A = (int *)malloc(M * K * sizeof(int));
    h_B = (int *)malloc(K * N * sizeof(int));
    h_C = (int *)malloc(M * N * sizeof(int));

    std::cout << "Generating input matrices..." << std::endl;
    generateMatrix(h_A, M, K);
    generateMatrix(h_B, K, N);

    int *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(int)));

    std::cout << "Copying input matrices to device..." << std::endl;
    HIP_CHECK(hipMemcpy(d_A, h_A, M * K * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, K * N * sizeof(int), hipMemcpyHostToDevice));

    // Grid and block size
    dim3 block(TILE_SIZE, TILE_SIZE, 1); // 8 x 8 threads per block = 64 threads per block
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);

    std::cout << "Launching kernel..." << std::endl;

    hipLaunchKernelGGL(
        matrixMultiplication, 
        grid, 
        block, 
        0, 
        0, 
        d_A, d_B, d_C, WIDTH
    );
    
    
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Copying output matrix to host..." << std::endl;
    HIP_CHECK(hipMemcpy(h_C, d_C, M * N * sizeof(int), hipMemcpyDeviceToHost));

    free(h_A);
    free(h_B);
    free(h_C);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;

}

void generateMatrix(int *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {   
        matrix[i] = i;
    }
}