#!/bin/bash

# Benchmark script to compare all reduction implementations with 100M elements
# Usage: ./benchmark_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
N=100000000

echo "========================================"
echo "Reduction Benchmark Suite"
echo "Testing with N = $N elements (100 million)"
echo "========================================"
echo ""

# Array to store results
declare -a names
declare -a cpu_times
declare -a gpu_times
declare -a speedups
declare -a results
declare -a errors
declare -a statuses

# Build all examples
echo "Building all reduction examples..."
cd "${BUILD_DIR}"
make -j$(nproc) \
    01_reduction_atomic \
    02_reduction_shared \
    03_reduction_sequential \
    04_reduction_contiguous \
    05_warp_reduce \
    06_grid_reduction 2>&1 | grep -E "(Building|Linking|error)" || true
echo ""

# Function to run and parse results
run_benchmark() {
    local name=$1
    local executable="${BUILD_DIR}/reduction/${name}"
    
    if [ ! -f "$executable" ]; then
        echo "Error: $executable not found"
        return 1
    fi
    
    echo "Running ${name}..."
    output=$($executable $N)
    echo "$output"
    
    # Parse output: N=100000000, CPU=123.456ms, GPU=7.890ms, Speedup=15.65x, Result=0.123456, Error=1.23e-05, PASS
    cpu_time=$(echo "$output" | grep -oP 'CPU=\K[0-9.]+')
    gpu_time=$(echo "$output" | grep -oP 'GPU=\K[0-9.]+')
    speedup=$(echo "$output" | grep -oP 'Speedup=\K[0-9.]+')
    result=$(echo "$output" | grep -oP 'Result=\K[0-9.-]+')
    error=$(echo "$output" | grep -oP 'Error=\K[0-9.e+-]+')
    status=$(echo "$output" | grep -oP '(PASS|FAIL)$')
    
    names+=("$name")
    cpu_times+=("$cpu_time")
    gpu_times+=("$gpu_time")
    speedups+=("$speedup")
    results+=("$result")
    errors+=("$error")
    statuses+=("$status")
}

# Run all benchmarks
run_benchmark "01_reduction_atomic"
echo ""
run_benchmark "02_reduction_shared"
echo ""
run_benchmark "03_reduction_sequential"
echo ""
run_benchmark "04_reduction_contiguous"
echo ""
run_benchmark "05_warp_reduce"
echo ""
run_benchmark "06_grid_reduction"
echo ""

# Print comparison table
echo "========================================"
echo "Performance Comparison Table"
echo "========================================"
printf "%-25s | %10s | %10s | %8s | %8s | %s\n" "Solution" "CPU (ms)" "GPU (ms)" "Speedup" "Error" "Status"
echo "-----------------------------------------------------------------------------"

for i in "${!names[@]}"; do
    printf "%-25s | %10.3f | %10.3f | %8.2fx | %8.2e | %s\n" \
        "${names[$i]}" \
        "${cpu_times[$i]}" \
        "${gpu_times[$i]}" \
        "${speedups[$i]}" \
        "${errors[$i]}" \
        "${statuses[$i]}"
done

echo ""
echo "========================================"

# Find fastest solution
fastest_idx=0
fastest_time=${gpu_times[0]}
for i in "${!gpu_times[@]}"; do
    if (( $(echo "${gpu_times[$i]} < $fastest_time" | bc -l) )); then
        fastest_time=${gpu_times[$i]}
        fastest_idx=$i
    fi
done

echo "🏆 Fastest: ${names[$fastest_idx]} (${gpu_times[$fastest_idx]}ms)"
echo "========================================"
