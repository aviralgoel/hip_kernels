#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <utility>

// Template alias for creating SIMD vector types using compiler extension.
// Creates a vector of 'Rank' elements of type 'T' that maps to GPU/CPU SIMD registers.
template <typename T, uint32_t Rank>
using VecT = T __attribute__((ext_vector_type(Rank)));

// Compile-time utility to get the number of elements in a SIMD vector type.
// Returns the Rank (element count) of the vector type as a compile-time constant.
template <typename T, uint32_t Rank>
static constexpr int32_t vectorSize(VecT<T, Rank> const&v)
{
    return Rank;
}

using float16_t = _Float16;

// Define a load function for input A blocks:
// Size: (BLOCK_M x BLOCK_K)
// ASSUMPTION:
// - We want contiguous BLOCK_M sized column neighbors in register.
// - Data is in col_major format
// This means:
// - From A we will load K columns of size BLOCK_M to satisfy our input data
template <typename AFragT, uint32_t BLOCK_M>
__device__ AFragT load_A_16x16_col_major(float16_t const* input, int ld)
{
    // Here we want to load a 16x16 block of data.
    // Register Mapping:

    // Size              |   BLOCK_M  |   BLOCK_M   |   BLOCK_M   |   BLOCK_M    |  Vector
    // Register Element  | 0  ...  15 | 16  ...  31 | 32  ...  47 | 48  ...   63 |  Element
    //                    ____________ _____________ _____________ ______________
    // Reg 0 [0 :15]     |     K0     |     K4      |     K8      |     K12      |  v[0]
    // Reg 0 [16:31]     |     K1     |     K5      |     K9      |     K13      |  v[1]
    // Reg 1 [0 :15]     |     K2     |     K6      |     K10     |     K14      |  v[2]
    // Reg 1 [16:31]     |     K3     |     K7      |     K11     |     K15      |  v[3]

    static constexpr uint32_t VW = vectorSize(AFragT{});
    static constexpr uint32_t Dim = BLOCK_M;

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 4 elements.
    // We need to know where they start, and where the next elements are.
    auto startCoord2D = std::make_pair(threadIdx.x % Dim,         // Row
                                       (threadIdx.x / Dim) * VW); // Col
    auto stepCoord2D = std::make_pair(0u, 1u);

    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

    auto startOffset = col_major(startCoord2D, ld);
    auto kOffset = col_major(stepCoord2D, ld);

    // If you notice carefully, kOffset != 1.
    // This means the following is vector is loaded with 4 non-contiguous offsets,
    // which the compiler will separate into 4 different global_load_short instructions.
    auto fragA = AFragT
    {
        input[startOffset],               // v[0] = Reg 0 [0:15]
        input[startOffset + kOffset],     // v[1] = Reg 0 [16:31]
        input[startOffset + 2 * kOffset], // v[2] = Reg 1 [0:15]
        input[startOffset + 3 * kOffset], // v[3] = Reg 1 [16:31]
    };

    return fragA;
}

// Define a load function for input B blocks:
// Size: (BLOCK_K x BLOCK_N)
// ASSUMPTION:
// - We want contiguous BLOCK_N sized row neighbors in register.
// - Data is in row_major format
// This means:
// - From B we will load K rows of size BLOCK_N to satisfy our input data
template <typename BFragT, uint32_t BLOCK_N>
__device__ BFragT load_B_16x16_row_major(float16_t const* input, int ld)
{
    // Here we want to load a 16x16 block of data.
    // Register Mapping:

    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N    |  Vector
    // Register Element  | 0  ...  15 | 16  ...  31 | 32  ...  47 | 48  ...   63 |  Element
    //                    ____________ _____________ _____________ ______________
    // Reg 0 [0 :15]     |     K0     |     K4      |     K8      |     K12      |  v[0]
    // Reg 0 [16:31]     |     K1     |     K5      |     K9      |     K13      |  v[1]
    // Reg 1 [0 :15]     |     K2     |     K6      |     K10     |     K14      |  v[2]
    // Reg 1 [16:31]     |     K3     |     K7      |     K11     |     K15      |  v[3]

    static constexpr uint32_t VW = vectorSize(BFragT{});
    static constexpr uint32_t Dim = BLOCK_N;

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 4 elements.
    // We need to know where they start, and where the next elements are.
    auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                        threadIdx.x % Dim);      // Col
    auto stepCoord2D = std::make_pair(1u, 0u);

    // Flatten to 1D row_major offsets.
    auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

    auto startOffset = row_major(startCoord2D, ld);
    auto kOffset = row_major(stepCoord2D, ld);

    // If you notice carefully, kOffset != 1.
    // This means the following is vector is loaded with 4 non-contiguous offsets,
    // which the compiler will separate into 4 different global_load_short instructions.
    auto fragB = BFragT
    {
        input[startOffset],               // v[0] = Reg 0 [0:15]
        input[startOffset + kOffset],     // v[1] = Reg 0 [16:31]
        input[startOffset + 2 * kOffset], // v[2] = Reg 1 [0:15]
        input[startOffset + 3 * kOffset], // v[3] = Reg 1 [16:31]
    };

    return fragB;
}

