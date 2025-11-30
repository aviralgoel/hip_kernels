#include <iostream>
#include <chrono>
#include "../utility/hip_utility.hpp"
#include "../utility/ops_gemm.hpp"

#define FILTER_SIZE 5
__constant__ float filter_constant[FILTER_SIZE * FILTER_SIZE];

__global__ void convolutionKernel(float* input, float* output, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int i = 0; i < FILTER_SIZE; i++)
        {
            for (int j = 0; j < FILTER_SIZE; j++)
            {
                float filter_value = filter_constant[i * FILTER_SIZE + j];
                int input_row = row + (i - FILTER_SIZE / 2);
                int input_col = col + (j - FILTER_SIZE / 2);
                if (input_row >= 0 && input_row < M && input_col >= 0 && input_col < N)
                {
                    sum += input[input_row * N + input_col] * filter_value;
                }
            }
        }
        output[row * N + col] = sum;
    }
}

void cpu_convolution(float* input, float* filter, float* output, int M, int N)
{
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < FILTER_SIZE; i++)
            {
                for (int j = 0; j < FILTER_SIZE; j++)
                {
                    float filter_value = filter[i * FILTER_SIZE + j];
                    int input_row = row + (i - FILTER_SIZE / 2);
                    int input_col = col + (j - FILTER_SIZE / 2);
                    
                    // Check boundary conditions (zero-padding)
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

int main() {


    int M = 5000;
    int N = 1000;

    float* input_host = new float[M * N];
    float* output_host = new float[M * N];
    float* filter_host = new float[FILTER_SIZE * FILTER_SIZE];

    // initialize the input
    float value = 1.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            input_host[i * N + j] = value;
            value += 1.0f;
        }
    }
    //print_matrix(input_host, M, N);

    // initialize the filter
    value = 1.0f;
        for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            filter_host[i * FILTER_SIZE + j] = value;
            value += 0.1f;
        }   
    }
    //print_matrix(filter_host, FILTER_SIZE, FILTER_SIZE);

    // allocate memory for the matrices on the device
    float* input_device;
    float* output_device;
    HIP_CHECK(hipMalloc(&input_device, M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&output_device, M * N * sizeof(float)));

    // copy the input and filter from the host to the device
    HIP_CHECK(hipMemcpyToSymbol(filter_constant, filter_host, FILTER_SIZE* FILTER_SIZE * sizeof(float)));
    HIP_CHECK(hipMemcpy(input_device, input_host, M * N * sizeof(float), hipMemcpyHostToDevice));

    // set the block and grid size
    dim3 block(4, 3); // X, Y dimensions of the block
    std::cout << "block x: " << block.x << ", block y: " << block.y << std::endl;
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    // print grid x and y
    std::cout << "grid x: " << grid.x << ", grid y: " << grid.y << std::endl;
    printf("number of blocks: %d\n", grid.x * grid.y);
    printf("number of threads per block: %d\n", block.x * block.y);

    // Create HIP events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Record start event
    HIP_CHECK(hipEventRecord(start, 0));

    // launch the kernel
    hipLaunchKernelGGL(convolutionKernel, grid, block, 0, 0, input_device, output_device, M, N);

    // Record stop event
    HIP_CHECK(hipEventRecord(stop, 0));
    
    // CPU convolution for comparison
    float* output_host_reference = new float[M * N];
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_convolution(input_host, filter_host, output_host_reference, M, N);
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
    
    HIP_CHECK(hipMemcpy(output_host, output_device, M * N * sizeof(float), hipMemcpyDeviceToHost));
    //print_matrix(output, M, N);
    std::cout << "Kernel execution completed" << std::endl;

    // compare the result with the reference result
    compare_float_matrices(output_host, output_host_reference, M, N, 64);

    // free the memory on the device
    HIP_CHECK(hipFree(input_device));
    HIP_CHECK(hipFree(output_device));

    // Clean up events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // free the memory on the host
    delete[] input_host;
    delete[] filter_host;
    delete[] output_host;
    delete[] output_host_reference;
    
    return 0;
}