#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <omp.h>  // For OpenMP functions

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../utility/hip_utility.hpp"
#include "../utility/ops_gemm.hpp"  // For fillRand utility function
#include "../include/math.hpp"  // For ceilDiv utility function
#include "../include/mfma_utility.hpp"  // For MFMA load functions

/////////////
// Helpers //
/////////////


// GPU device function to fill all elements of a SIMD vector with the same value.
// Broadcasts a scalar value to all Rank elements of the vector.
// Usage: vectorFill(my_float4_vec, 0.0f); // Sets all 4 elements to 0.0f
template <typename T, uint32_t Rank>
__device__ void vectorFill(VecT<T, Rank> &v, T value)
{
    for (uint32_t i = 0; i < Rank; i++)
    {
        v[i] = value;
    }
}

///////////////
// Constants // 
//////////////

const int WAVE_SIZE = 64; // Number of threads in a wavefront

const int T_BLOCK_X = 1 * WAVE_SIZE; // Number of threads in the X dimension of the block
const int T_BLOCK_Y = 1; // Number of threads in the Y dimension of the block

const int BLOCK_M = 16; // MFMA block size in the M dimension
const int BLOCK_N = 16; // MFMA block size in the N dimension
const int BLOCK_K = 16; // MFMA block size in the K dimension

///////////
// Types //
///////////

using float16_t = _Float16;  // C++ standard half-precision floating-point type
using float32_t = float;

// Fragments (per thread vector) for MFMA.
using AFragT = VecT<float16_t, BLOCK_M * BLOCK_K / WAVE_SIZE>; // Fragment of A matrix for MFMA.
using BFragT = VecT<float16_t, BLOCK_K * BLOCK_N / WAVE_SIZE>; // Fragment of B matrix for MFMA.
using AccumFragT = VecT<float32_t, BLOCK_M * BLOCK_N / WAVE_SIZE>; // Fragment of accumulator for MFMA.
using CFragT = VecT<float16_t, BLOCK_M * BLOCK_N / WAVE_SIZE>; // Fragment of C matrix for MFMA.

// wrapper for the MFMA instruction.
__device__ AccumFragT mfma_f32_16x16x16_f16(AFragT afr, BFragT bfr, AccumFragT accumfr)
{
    return __builtin_amdgcn_mfma_f32_16x16x16f16(afr, bfr, accumfr, 0, 0, 0);
}


__global__ void sgemm_example_d(uint32_t     m,
                                uint32_t     n,
                                uint32_t     k,
                                float16_t const* a,
                                float16_t const* b,
                                float32_t const* c,
                                float32_t*       d,
                                uint32_t     lda,
                                uint32_t     ldb,
                                uint32_t     ldc,
                                uint32_t     ldd,
                                float32_t        alpha,
                                float32_t        beta)
{
    AFragT fragA = AFragT{};
    BFragT fragB = BFragT{};
    AccumFragT fragAcc = AccumFragT{};

    vectorFill(fragAcc, 0.0f); 
    
    auto waveGridX = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
    auto waveGridY = (blockIdx.y * blockDim.y + threadIdx.y);

    auto cRow = waveGridX * BLOCK_M;
    auto cCol = waveGridY * BLOCK_N;

    if (cRow < m && cCol < n) 
    {
        ///
        /// Step 1: accumulate A x B by stepping through k dimension
        ///
        for(int i = 0; i < k; i += BLOCK_K)
        {
            // Load the inputs.
            // Flatten 2D coord (row, col) into 1D, knowing:
            // A = col major, BLOCK_M x BLOCK_K
            // B = row major, BLOCK_K x BLOCK_N
            fragA = load_A_16x16_col_major<AFragT, BLOCK_M>(a + (cRow  + i * lda), lda);
            fragB = load_B_16x16_row_major<BFragT, BLOCK_N>(b + (i * ldb + cCol), ldb);

            // // Matrix multiply-accumulate using MFMA units
            // // Accumulation intermediate = BLOCK_M x BLOCK_N
            // fragAcc = mfma_f32_16x16x16f16(fragA, fragB, fragAcc);
        }
    }
}



////////////
// Functions //
/////////////   
void gemm_mfma(int M, int N, int K, float alpha, float beta);

int main(int argc, char* argv[]) {
    
    // Default values
    int M = 16;
    int N = 16;
    int K = 32;
    float alpha = 1.0f;
    float beta = 0.0f;

    // Parse command line arguments
    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [M] [N] [K]" << std::endl;
            std::cout << "  M : Number of rows in A and C (default: 32)" << std::endl;
            std::cout << "  N : Number of columns in B and C (default: 16)" << std::endl;
            std::cout << "  K : Number of columns in A / rows in B (default: 32)" << std::endl;
            std::cout << "\nNote: M, N, K must be multiples of 16" << std::endl;
            std::cout << "Alpha = 1.0, Beta = 0.0 (fixed)" << std::endl;
            return 0;
        }
        M = std::atoi(argv[1]);
    }
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) K = std::atoi(argv[3]);

    gemm_mfma(M, N, K, alpha, beta);

    return 0;
}

void gemm_mfma(int M, int N, int K, float alpha, float beta)
{
    // Bounds check
    if((M < (BLOCK_M * T_BLOCK_X / WAVE_SIZE) 
       || N < (BLOCK_N * T_BLOCK_Y) 
       || K < BLOCK_K)
       || (M % BLOCK_M || N % BLOCK_N || K % BLOCK_K))
    {
        std::cout << "Unsupported size!\n";
        return;
    }

    // Leading dimensions
    uint32_t lda = M; // leading dimension of A (col major)
    uint32_t ldb = N; // leading dimension of B (row major)
    uint32_t ldc = M; // leading dimension of C (col major)
    uint32_t ldd = M; // leading dimension of D (col major)


    std::vector<float16_t> A(M * K);
    std::vector<float16_t> B(K * N);
    std::vector<float32_t> C(M * N);
    std::vector<float32_t> D(M * N, std::numeric_limits<float32_t>::signaling_NaN());

    fillRand(A.data(), M, K);
    fillRand(B.data(), K, N);
    fillRand(C.data(), M, N);

    std::cout << "Initializing host data..." << std::endl;
    std::cout << "Matrix dimensions: A(" << M << "x" << K << ") × B(" << K << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Beta: " << beta << std::endl;
    std::cout << "Leading dimensions: LDA(" << lda << "), LDB(" << ldb << "), LDC(" << ldc << "), LDD(" << ldd << ")" << std::endl;

    float16_t* d_A;
    float16_t* d_B;
    float32_t* d_C;
    float32_t* d_D;  // Output matrix D is float32, not float16

    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float32_t)));
    HIP_CHECK(hipMalloc(&d_D, M * N * sizeof(float32_t)));

    HIP_CHECK(hipMemcpy(d_A, A.data(), M * K * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B.data(), K * N * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, C.data(), M * N * sizeof(float32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_D, D.data(), M * N * sizeof(float32_t), hipMemcpyHostToDevice));

    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(ceilDiv(M, BLOCK_M * T_BLOCK_X / WAVE_SIZE),
                         ceilDiv(N, BLOCK_N * T_BLOCK_Y));

    std::cout << "Launching GEMM kernel..." << std::endl;
    std::cout << "TBlock(X, Y) = (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    std::cout << "GridDim(X, Y) = (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;

    // Create events for timing
    hipEvent_t startEvent, stopEvent;
    HIP_CHECK(hipEventCreate(&startEvent));
    HIP_CHECK(hipEventCreate(&stopEvent));

    // Launch kernel with timing
    hipExtLaunchKernelGGL(sgemm_example_d,
                          gridDim,
                          blockDim,
                          0,           // sharedMemBytes
                          0,           // stream
                          startEvent,  // event start
                          stopEvent,   // event stop
                          0,           // flags
                          static_cast<uint32_t>(M),
                          static_cast<uint32_t>(N),
                          static_cast<uint32_t>(K),
                          d_A,
                          d_B,
                          d_C,
                          d_D,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          alpha,
                          beta);

    // Get timing results
    auto elapsedTimeMs = 0.0f;
    HIP_CHECK(hipEventSynchronize(stopEvent));
    HIP_CHECK(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    HIP_CHECK(hipEventDestroy(startEvent));
    HIP_CHECK(hipEventDestroy(stopEvent));

    // GEMM flops converge to 2 * mnk
    auto gFlops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) * 1.0e-9;
    auto tFlopsPerSec = gFlops / static_cast<double>(elapsedTimeMs * 1e-3);

    // Echo performance
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "BlkM: " << BLOCK_M << ", BlkN: " << BLOCK_N << ", BlkK: " << BLOCK_K << std::endl
              << "MatM: " << M << ", MatN: " << N << ", MatK: " << K << std::endl
              << "alpha: " << alpha << ", lda: " << lda << ", ldb: " << ldb << std::endl
              << "beta: " << beta << ", ldc: " << ldc << ", ldd: " << ldd << std::endl
              << "elapsedMs: " << elapsedTimeMs << ", Problem Size(GFlops): " << gFlops << ", TFlops/s: " << tFlopsPerSec << std::endl;

    std::cout << "\nValidating result with reference..." << std::endl;
}
