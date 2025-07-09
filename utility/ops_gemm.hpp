#pragma once

#include <iostream>
#include <cstdlib>
#include <cstdio>

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
            C[i * N + j] = 0;
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