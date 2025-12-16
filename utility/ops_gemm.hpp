#pragma once

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <type_traits>
#include <omp.h>
#include <cmath>
#include <mutex>

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

struct row_major{};
struct col_major{};

// Element-wise comparison
template <typename DataT>
__host__ std::pair<bool, double>
         compareEqual(DataT const* a, DataT const* b, uint32_t size, double tolerance = 10.0)
{
    bool   retval             = true;
    double max_relative_error = 0.0;

    // Some types don't have direct conversion to double.
    // Convert to float first then to double.
    auto toDouble = [](DataT const& val) { return static_cast<double>(static_cast<float>(val)); };

    bool       isInf = false;
    bool       isNaN = false;
    std::mutex writeMutex;

#pragma omp parallel for
    for(int i = 0; i < size; ++i)
    {
        auto valA = a[i];
        auto valB = b[i];

        auto numerator = fabs(toDouble(valA) - toDouble(valB));
        auto divisor   = fabs(toDouble(valA)) + fabs(toDouble(valB)) + 1.0;

        if(std::isinf(numerator) || std::isinf(divisor))
        {
#pragma omp atomic
            isInf |= true;
        }
        else
        {
            auto relative_error = numerator / divisor;
            if(std::isnan(relative_error))
            {
#pragma omp atomic
                isNaN |= true;
            }
            else if(relative_error > max_relative_error)
            {
                const std::lock_guard<std::mutex> guard(writeMutex);
                // Double check in case of stall
                if(relative_error > max_relative_error)
                {
                    max_relative_error = relative_error;
                }
            }
        }

        if(isInf || isNaN)
        {
            i = size;
        }
    }

    auto eps = toDouble(std::numeric_limits<DataT>::epsilon());
    if(isInf)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::infinity();
    }
    else if(isNaN)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::signaling_NaN();
    }
    else if(max_relative_error > (eps * tolerance))
    {
        retval = false;
    }

    return std::make_pair(retval, max_relative_error);
}

// Host GEMM validation
template <typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD = LayoutC>
__host__ void gemm_cpu_h(uint32_t       m,
                         uint32_t       n,
                         uint32_t       k,
                         InputT const*  a,
                         InputT const*  b,
                         OutputT const* c,
                         OutputT*       d,
                         uint32_t       lda,
                         uint32_t       ldb,
                         uint32_t       ldc,
                         uint32_t       ldd,
                         ComputeT       alpha,
                         ComputeT       beta)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto aIndex = std::is_same<LayoutA, row_major>::value ? rowMjr : colMjr;
    auto bIndex = std::is_same<LayoutB, row_major>::value ? rowMjr : colMjr;
    auto cIndex = std::is_same<LayoutC, row_major>::value ? rowMjr : colMjr;
    auto dIndex = std::is_same<LayoutD, row_major>::value ? rowMjr : colMjr;

#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
#pragma omp parallel for
        for(int j = 0; j < n; ++j)
        {
            ComputeT accum = static_cast<ComputeT>(0);
            for(int h = 0; h < k; ++h)
            {
                accum += static_cast<ComputeT>(a[aIndex(i, h, lda)])
                         * static_cast<ComputeT>(b[bIndex(h, j, ldb)]);
            }
            d[dIndex(i, j, ldd)] = static_cast<OutputT>(
                alpha * accum + beta * static_cast<ComputeT>(c[cIndex(i, j, ldc)]));
        }
    }
}