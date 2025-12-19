#include <iostream>
#include <chrono>
#include "../utility/ops_gemm.hpp"
#include "../utility/hip_utility.hpp"

// Tiled convolution: each block loads INPUT_TILE_SIZE elements (including halo)
// and produces OUTPUT_TILE_SIZE valid outputs
#define FILTER_SIZE 5
#define FILTER_RADIUS (FILTER_SIZE / 2)
#define INPUT_TILE_SIZE 32
#define OUTPUT_TILE_SIZE (INPUT_TILE_SIZE - 2 * FILTER_RADIUS)

__constant__ float filter_constant[FILTER_SIZE * FILTER_SIZE];

// CPU reference implementation for validation
void cpu_convolution(float* input, float* filter, float* output, int M, int N, int filter_size)
{
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < filter_size; i++)
            {
                for (int j = 0; j < filter_size; j++)
                {
                    float filter_value = filter[i * filter_size + j];
                    int input_row = row + (i - filter_size / 2);
                    int input_col = col + (j - filter_size / 2);
                    
                    if (input_row >= 0 && input_row < M && input_col >= 0 && input_col < N)
                    {
                        sum += input[input_row * N + input_col] * filter_value;
                    }
                }
            }
            output[row * N + col] = sum;
        }
    }
}

__global__ void tiled_convolutionKernel(float* input, float* output, int M, int N)
{
    // Calculate input position (including halo region for boundary elements)
    int inputRow = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - FILTER_RADIUS;
    int inputCol = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - FILTER_RADIUS;

    __shared__ float input_lds[INPUT_TILE_SIZE][INPUT_TILE_SIZE];
    
    // Load input tile into shared memory (all threads participate)
    if (inputRow >= 0 && inputCol >= 0 && inputRow < M && inputCol < N)
    {
        input_lds[threadIdx.y][threadIdx.x] = input[inputRow * N + inputCol];
    }
    else
    {
        input_lds[threadIdx.y][threadIdx.x] = 0.0f;  // Zero-padding for out-of-bounds
    }
    
    __syncthreads();

    // Calculate tile position (only center threads compute output)
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileCol = threadIdx.x - FILTER_RADIUS;
    
    // Only threads that can apply full filter contribute to output
    if (tileRow >= 0 && tileCol >= 0 && tileRow < OUTPUT_TILE_SIZE && tileCol < OUTPUT_TILE_SIZE)
    {
        int outputRow = blockIdx.y * OUTPUT_TILE_SIZE + tileRow;
        int outputCol = blockIdx.x * OUTPUT_TILE_SIZE + tileCol;
        
        if (outputRow >= 0 && outputRow < M && outputCol >= 0 && outputCol < N)
        {
            // Apply filter using shared memory data
            float sum = 0.0f;
            for (int i = 0; i < FILTER_SIZE; i++)
            {
                for (int j = 0; j < FILTER_SIZE; j++)
                {
                    sum += input_lds[tileRow + i][tileCol + j] * filter_constant[i * FILTER_SIZE + j];
                }
            }
            output[outputRow * N + outputCol] = sum;
        }
    }
}

int main()
{
    int M = 5000;
    int N = 1000;

    // Allocate and initialize host memory
    float* input_host = new float[M * N];
    float* output_host = new float[M * N];
    float* filter_host = new float[FILTER_SIZE * FILTER_SIZE];

    float value = 1.0f;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            input_host[i * N + j] = value;
            value += 1.0f;
        }
    }

    value = 1.0f;
    for (int i = 0; i < FILTER_SIZE; i++)
    {
        for (int j = 0; j < FILTER_SIZE; j++)
        {
            filter_host[i * FILTER_SIZE + j] = value;
            value += 0.1f;
        }
    }

    // Allocate device memory
    float* input_device;
    float* output_device;
    HIP_CHECK(hipMalloc(&input_device, M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&output_device, M * N * sizeof(float)));

    // Copy data to device (filter goes to constant memory)
    HIP_CHECK(hipMemcpy(input_device, input_host, M * N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyToSymbol(filter_constant, filter_host, FILTER_SIZE * FILTER_SIZE * sizeof(float)));

    // Block size = input tile size, grid covers output domain
    dim3 block(INPUT_TILE_SIZE, INPUT_TILE_SIZE);
    dim3 grid((N + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, 
              (M + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE);
    
    printf("Block: (%d, %d)\n", block.x, block.y);
    printf("Grid: (%d, %d)\n", grid.x, grid.y);
    printf("Total blocks: %d\n", grid.x * grid.y);
    printf("Threads per block: %d\n", block.x * block.y);

    // GPU execution with event timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, 0));
    
    hipLaunchKernelGGL(tiled_convolutionKernel, grid, block, 0, 0, input_device, output_device, M, N);
    
    HIP_CHECK(hipEventRecord(stop, 0));
    
    // CPU reference (runs while GPU is executing)
    float* output_host_reference = new float[M * N];
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_convolution(input_host, filter_host, output_host_reference, M, N, FILTER_SIZE);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Calculate and display timings
    float gpu_time_ms;
    HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, start, stop));
    auto cpu_time_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    printf("GPU Time: %.3f ms\n", gpu_time_ms);
    printf("CPU Time: %.3f ms\n", cpu_time_ms);
    printf("Speedup: %.2fx\n", cpu_time_ms / gpu_time_ms);
    
    // Validate results
    HIP_CHECK(hipMemcpy(output_host, output_device, M * N * sizeof(float), hipMemcpyDeviceToHost));
    compare_float_matrices(output_host, output_host_reference, M, N, 64);

    // Cleanup
    HIP_CHECK(hipFree(input_device));
    HIP_CHECK(hipFree(output_device));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    delete[] input_host;
    delete[] filter_host;
    delete[] output_host;
    delete[] output_host_reference;
    
    return 0;
}