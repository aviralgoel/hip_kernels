#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/////////////
// Helpers //
/////////////

// Template alias for creating SIMD vector types using compiler extension.
// Creates a vector of 'Rank' elements of type 'T' that maps to GPU/CPU SIMD registers.
// Usage: VecT<float, 4> my_float4_vec; // Creates a vector of 4 float elements.
template <typename T, uint32_t Rank>
using VecT = T __attribute__((ext_vector_type(Rank)));

// Compile-time utility to get the number of elements in a SIMD vector type.
// Returns the Rank (element count) of the vector type as a compile-time constant.
// Usage: auto size = vectorSize(my_float4_vec); // returns 4
template <typename T, uint32_t Rank>
static constexpr int32_t vectorSize(VecT<T, Rank> const&v)
{
    return Rank;
}

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

const int WAVE_SIZE = 64; // Number of threads in a wavefront.

const int T_BLOCK_X = 1 * WAVE_SIZE;
const int T_BLOCK_Y = 1;

const int BLOCK_M = 16; // MFMA block size in the M dimension.
const int BLOCK_N = 16; // MFMA block size in the N dimension.
const int BLOCK_K = 16; // MFMA block size in the K dimension.

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


////////////
// Functions //
/////////////   
void gemm_mfma(int M, int N, int K, float alpha, float beta);

int main() {
    
    int M = 16;
    int N = 16;
    int K = 32;
    float alpha = 1.0f;
    float beta = 0.0f;

    gemm_mfma(M, N, K, alpha, beta);

    return 0;
}

void gemm_mfma(int M, int N, int K, float alpha, float beta)
{
    std::cout << "Running MFMA GEMM..." << std::endl;
    std::cout << "Matrix dimensions: A(" << M << "x" << K << ") × B(" << K << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Beta: " << beta << std::endl;

    // allocate memory for the matrices
    float16_t *A = new float16_t[M * K];
    float16_t *B = new float16_t[K * N];
    float32_t *C = new float32_t[M * N];
}

