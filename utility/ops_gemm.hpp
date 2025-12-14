#pragma once

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <type_traits>
#include <omp.h>

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

void initialize_matrix(float *matrix, int M, int N)
{
    printf("Initializing matrix...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 9.0f + 1.0f;  
        }
    }
}

void print_matrix(float *matrix, int M, int N)
{
    printf("Printing matrix...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::fixed << std::setprecision(2) << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}
void cpu_gemm(int *A, int *B, int *C, int M, int N, int K)
{   
    printf("Running CPU GEMM...\n");
    int progress_interval = M / 20;  // Update every 5% (20 intervals = 5% each)
    if (progress_interval == 0) progress_interval = 1;  // For small matrices
    
    for (int i = 0; i < M; i++) {
        // Show progress
        if (i % progress_interval == 0 || i == M - 1) {
            int percent = (int)((float)(i + 1) * 100.0f / M);
            printf("\rCPU Progress: %d%% (%d/%d rows)", percent, i + 1, M);
            fflush(stdout);  // Force immediate output
        }
        
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
    printf("\n");  // New line after completion
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

void compare_float_matrices(float* gpu_result, float* cpu_result, int M, int N, float tolerance = 1e-6f) {
    printf("Comparing matrices...\n");
    int error_count = 0;
    float max_diff = 0.0f;
    
    // for each element in the matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            
            float diff = fabs(gpu_result[i * N + j] - cpu_result[i * N + j]);
            
            // max difference for the whole matrix
            if (diff > max_diff) {
                max_diff = diff;
            }
            
            // is this error acceptable?
            if (diff > tolerance) {

                // print the first 10 errors
                if (error_count < 10) {
                    printf("Error at position (%d, %d): GPU = %.6f, CPU = %.6f, diff = %.9f\n", 
                           i, j, gpu_result[i * N + j], cpu_result[i * N + j], diff);
                }
                error_count++;
            }
        }
    }
    
    if (error_count == 0) {
        printf("Matrices are equal (within tolerance %.9f)\n", tolerance);
        printf("Max difference: %.9f\n", max_diff);
    } else {
        printf("Matrices are not equal\n");
        printf("Error count: %d (%.2f%%)\n", error_count, (float)error_count / (M * N) * 100);
        printf("Max difference: %.9f\n", max_diff);
    }
}

// Host matrix data random initialization with OpenMP parallelization
template <typename DataT>
static inline void fillRand(DataT* mat, uint32_t m, uint32_t n)
{
    auto randInit = []() {
        srand(time(0));
        return 0u;
    };

    static auto init = randInit();
    
    #pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        auto rando = rand() % 5u;
        for(int j = 0; j < n; j++)
        {
            // Assign random integer values within 0-4, alternating
            // sign if the value is a multiple of 3
            auto value     = (rando + j) % 5u;
            mat[i * n + j] = ((value % 3u == 0u) && std::is_signed<DataT>::value)
                                 ? -static_cast<DataT>(value)
                                 : static_cast<DataT>(value);
        }
    }
}