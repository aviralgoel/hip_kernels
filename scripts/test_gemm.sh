#!/bin/bash

echo "Starting comprehensive GEMM performance tests..."

# Build the project
cd build && cmake --build .

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful. Running tests..."

# Test both implementations if available
NAIVE_GEMM="./gemm/naive_gemm"
TILED_GEMM="./gemm/tiled_gemm"

echo "Available implementations:"
[ -x "$NAIVE_GEMM" ] && echo "  - Naive GEMM" || echo "  - Naive GEMM: NOT FOUND"
[ -x "$TILED_GEMM" ] && echo "  - Tiled GEMM" || echo "  - Tiled GEMM: NOT FOUND"
echo ""

# Helper function to run test with both implementations
run_test() {
    local test_name=$1
    local args=$2
    
    echo "--- $test_name ---"
    
    if [ -x "$NAIVE_GEMM" ]; then
        echo "Naive GEMM:"
        $NAIVE_GEMM $args
        echo ""
    fi
    
    if [ -x "$TILED_GEMM" ]; then
        echo "Tiled GEMM:"
        $TILED_GEMM $args
        echo ""
    fi
    
    echo "----------------------------------------"
}

# Edge case tests
echo "=== Edge Case Tests ==="
run_test "1×1 matrix" "1"
run_test "2×2 matrix" "2"
run_test "3×3 matrix" "3"
run_test "4×4 matrix" "4"
run_test "16×16 matrix (tile aligned)" "16"
run_test "17×17 matrix (tile misaligned)" "17"
run_test "15×15 matrix (tile misaligned)" "15"

# Quick correctness tests
echo "=== Quick Correctness Tests ==="
for size in 32 64 128 256; do
    run_test "${size}×${size} correctness" "$size"
done

# Rectangular matrix tests
echo "=== Rectangular Matrix Tests ==="
run_test "Tall matrix 512×128×256" "512 128 256"
run_test "Wide matrix 128×512×256" "128 512 256"
run_test "Skinny matrix 1024×32×64" "1024 32 64"
run_test "Fat matrix 32×1024×64" "32 1024 64"

# Non-square K dimension tests
echo "=== Non-Square K Dimension Tests ==="
run_test "Matrix 256×128×512" "256 128 512"
run_test "Matrix 128×256×512" "128 256 512"
run_test "Matrix 512×256×128" "512 256 128"

# Tile boundary tests (for tiled implementation)
echo "=== Tile Boundary Tests ==="
run_test "Multiple of 16: 160×160" "160"
run_test "Not multiple of 16: 100×100" "100"
run_test "Prime size: 97×97" "97"
run_test "Large prime: 251×251" "251"

# Performance benchmarks
echo "=== Performance Benchmarks ==="
for size in 512 1024 1536 2048; do
    run_test "${size}×${size} performance" "$size"
done

# Memory stress tests
echo "=== Memory Stress Tests ==="
run_test "Large square: 3072×3072" "3072"
run_test "Very large square: 4096×4096" "4096"

# Extreme aspect ratio tests
echo "=== Extreme Aspect Ratio Tests ==="
run_test "Very tall: 2048×64×128" "2048 64 128"
run_test "Very wide: 64×2048×128" "64 2048 128"
run_test "Extreme K: 256×256×4096" "256 256 4096"

# Power of 2 tests
echo "=== Power of 2 Tests ==="
for exp in 5 6 7 8 9 10; do
    size=$((2**exp))
    run_test "2^${exp} = ${size}×${size}" "$size"
done

# Implementation comparison (if both available)
if [ -x "$NAIVE_GEMM" ] && [ -x "$TILED_GEMM" ]; then
    echo "=== Implementation Comparison ==="
    echo "Comparing performance on various sizes..."
    
    for size in 256 512 1024 2048; do
        echo "Size: ${size}×${size}"
        echo "Naive:"
        $NAIVE_GEMM $size | grep -E "(GPU Time|CPU Time|Speedup)"
        echo "Tiled:"
        $TILED_GEMM $size | grep -E "(GPU Time|CPU Time|Speedup)"
        echo ""
    done
fi

# Error handling tests
echo "=== Error Handling Tests ==="
echo "Testing invalid arguments..."
if [ -x "$NAIVE_GEMM" ]; then
    echo "Naive GEMM with invalid args:"
    $NAIVE_GEMM 0 0 0 2>/dev/null || echo "  Handled gracefully"
    $NAIVE_GEMM -1 2>/dev/null || echo "  Handled gracefully"
    $NAIVE_GEMM abc 2>/dev/null || echo "  Handled gracefully"
fi

if [ -x "$TILED_GEMM" ]; then
    echo "Tiled GEMM with invalid args:"
    $TILED_GEMM 0 0 0 2>/dev/null || echo "  Handled gracefully"
    $TILED_GEMM -1 2>/dev/null || echo "  Handled gracefully"
    $TILED_GEMM abc 2>/dev/null || echo "  Handled gracefully"
fi

echo ""
echo "All tests completed!"
echo "Check the output above for any failures or performance issues." 