#include <hip/hip_runtime.h>
#include "../include/HIP/HIP_Utils.hpp"

#include <iostream>
#include <cstdlib>
#include <vector>

#define TILE_SIZE 32

void generateMatrix(float *matrix, int rows, int cols);

__global__ void naiveMatrixMultiplication(float *A, float *B, float *C, int N)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < N && y < N) 
    {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) 
        {
            sum += A[y * N + i] * B[i * N + x];
        }
        C[y * N + x] = sum;
    }
}

__global__ void tiledMatrixMultiplication(float *A, float *B, float* C, int M, int N, int K)
{
    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
    __shared__ float Nds[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int bx = blockIdx.x; 
    int by = blockIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    float Pvalue = 0;
    int totalPhases = (K + TILE_SIZE - 1) / TILE_SIZE;
    for(int ph = 0; ph < totalPhases; ph++)
    {
        Mds[ty][tx] = (Row < M && ph * TILE_SIZE + tx < K) ? A[Row * K + ph * TILE_SIZE + tx] : 0.0f;
        Nds[ty][tx] = (ph * TILE_SIZE + ty < K && Col < N) ? B[(ph * TILE_SIZE + ty) * N + Col] : 0.0f;
        
        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    if(Row < M && Col < N)
    {
        C[Row * N + Col] = Pvalue;
    }
}

int main()
{
    const int M = 10000, N = 10000, K = 10000;
    std::cout << "Matrix A: " << M << " x " << K << std::endl;
    std::cout << "Matrix B: " << K << " x " << N << std::endl;
    std::cout << "Matrix C: " << M << " x " << N << std::endl;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_tiledC(M * N);

    std::cout << "Generating input matrices..." << std::endl;
    generateMatrix(h_A.data(), M, K);
    generateMatrix(h_B.data(), K, N);

    float *d_A, *d_B, *d_C, *d_tiledC;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tiledC, M * N * sizeof(float)));

    std::cout << "Copying input matrices to device..." << std::endl;
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE, 1); // threads layout per block    
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1); // blocks layout per grid

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    std::cout << "Launching tiled kernel..." << std::endl;
    HIP_CHECK(hipEventRecord(start, 0));

    hipLaunchKernelGGL(
        tiledMatrixMultiplication, 
        grid, 
        block, 
        0, // shared memory size
        0, // stream
        d_A, d_B, d_tiledC, M, N, K
    );

    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsedTime;
    HIP_CHECK(hipEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Tiled kernel execution time: " << elapsedTime << " ms" << std::endl;

    std::cout << "Launching non-tiled kernel..." << std::endl;
    HIP_CHECK(hipEventRecord(start, 0));

    hipLaunchKernelGGL(
        naiveMatrixMultiplication, 
        grid, 
        block, 
        0, 
        0, 
        d_A, d_B, d_C, K
    );

    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    HIP_CHECK(hipEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Non-tiled kernel execution time: " << elapsedTime << " ms" << std::endl;

    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Copying output matrix to host..." << std::endl;
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_tiledC.data(), d_tiledC, M * N * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_tiledC));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return 0;
}

void generateMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {   
        matrix[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 10.0));
    }
}
