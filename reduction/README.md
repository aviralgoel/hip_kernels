# GPU Parallel Reduction: An Optimization Journey

This directory contains six progressively optimized implementations of parallel reduction on AMD GPUs using HIP. The goal: sum 100 million floating-point numbers as fast as possible.

## The Problem

Parallel reduction seems simple—just add numbers together. But on GPUs with thousands of threads, naive approaches create bottlenecks. Each solution here addresses a specific performance issue discovered in the previous one.

## Implementation Journey

### 01: Atomic Operations (Baseline)
**The approach:** Every thread atomically adds its element directly to the output.

```
Threads:  T0   T1   T2   T3   T4   T5   T6   T7
          |    |    |    |    |    |    |    |
Input:   [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]
          |    |    |    |    |    |    |    |
          └────┴────┴────┴────┴────┴────┴────┴──> atomicAdd(output)
                        ⚠️ CONTENTION
```

**The problem:** Atomic operations serialize access to memory. With millions of threads competing for the same memory location, this creates massive contention. Simple to code, but painfully slow.

### 02: Shared Memory with Interleaved Addressing
**The optimization:** Use shared memory for block-level reduction, then combine block results.

```
Step 1 (stride=1):  T0  T1  T2  T3  T4  T5  T6  T7
                    [0] [1] [2] [3] [4] [5] [6] [7]
                     +       +       +       +
                     |       |       |       |
Step 2 (stride=2):  [0]  X  [2]  X  [4]  X  [6]  X   (T1,T3,T5,T7 idle)
                     +           +
                     |           |
Step 3 (stride=4):  [0]  X   X   X  [4]  X   X   X   (divergence!)
                     +
                     |
Result:             [0]  ← final sum
```

**The problem:** Interleaved addressing (`threadIdx.x % (2*i) == 0`) causes two issues: divergent branches (threads within a warp take different paths) and bank conflicts (multiple threads access the same shared memory bank simultaneously).

### 03: Sequential Addressing
**The optimization:** Change addressing pattern to `index = 2*i*threadIdx.x` to eliminate divergent branches.

```
Step 1 (i=1):  T0  T1  T2  T3  T4  T5  T6  T7
               [0] [1] [2] [3] [4] [5] [6] [7]
                +   +   +   +
                |   |   |   |      (T4-T7 idle)
Step 2 (i=2):  [0] [2] [4] [6]  X   X   X   X
                +       +
                |       |          (T2,T3 idle)
Step 3 (i=4):  [0]     [4]  X   X   X   X   X
                +
                |                  (T1-T7 idle)
Result:        [0] ← final sum
```

**The problem:** While branches are less divergent, we still have bank conflicts. Half the threads go idle early, wasting computational resources.

### 04: Contiguous Addressing
**The optimization:** Reverse the loop direction. Adjacent threads now access adjacent memory locations (`sdata[threadIdx.x] += sdata[threadIdx.x + i]`).

```
Step 1 (i=4):  T0  T1  T2  T3  T4  T5  T6  T7
               [0] [1] [2] [3] [4] [5] [6] [7]
                +   +   +   +   ↑   ↑   ↑   ↑   (all threads active!)
                └───┴───┴───┴───┘
Step 2 (i=2):  [0] [1] [2] [3]
                +   +   ↑   ↑                   (T0-T1 active)
                └───┴───┘
Step 3 (i=1):  [0] [1]
                +   ↑                           (T0 active)
                └───┘
Result:        [0] ← final sum        ✓ Contiguous access pattern
```

**The benefit:** No bank conflicts, perfect memory coalescing, all threads active longer. This is the first "good" implementation.

### 05: Warp Shuffle
**The optimization:** Once we reach 32 elements, use warp shuffle intrinsics (`__shfl_down`) instead of shared memory.

```
Shared Memory Reduction: [256 elements] → [32 elements]
                              ↓
Warp Shuffle (no memory!):
  T0   T1   T2   T3  ...  T14  T15  T16 ... T31
  [0]  [1]  [2]  [3] ...  [14] [15] [16]... [31]
   +----+----+----+--------+----+              ← offset=16
  [0]  [1]  [2]  [3] ...  [14] [15]
   +----+----+                                 ← offset=8
  [0]  [1]  [2]  [3] ...
   +----+                                      ← offset=4,2,1
  [0] ← final sum (direct register exchange!)
```

**The benefit:** Shuffle operations communicate between threads in a warp without touching memory at all—lower latency, less bandwidth usage.

### 06: Grid-Stride Loop
**The optimization:** Each thread processes 8 elements before reducing. This shrinks the grid by 8x, reducing atomic operations and kernel launch overhead.

```
Each thread processes 8 elements before reduction:

Block 0:  T0                    T1                    T255
          [0  256  512 ... 1792] [1  257  513 ... 1793] ... [255 ... 2047]
           └──sum in register──┘  └──sum in register──┘      └──register──┘
                    |                      |                       |
          Shared Memory Reduction + Warp Shuffle
                              ↓
                        atomicAdd(output)  ← 8x fewer atomics!

Previous solutions: 1 thread → 1 element → N/256 blocks
This solution:      1 thread → 8 elements → N/2048 blocks  (87.5% reduction!)
```

**The benefit:** Fewer blocks means fewer atomic operations at the end. Better memory access patterns. Single kernel launch instead of iterative reductions.

## Performance Comparison

Run the benchmark to see results on your hardware:

```bash
cd /home/avirgoel/aviral/hip_kernels
mkdir -p build && cd build
cmake ..
make
cd ../reduction
./benchmark_all.sh
```

Typical results on MI300X (100M elements):
- **Solution 1 (Atomic):** ~150ms - Baseline
- **Solution 2 (Interleaved):** ~8ms - 18.75x faster
- **Solution 3 (Sequential):** ~7ms - 21.4x faster  
- **Solution 4 (Contiguous):** ~4ms - 37.5x faster
- **Solution 5 (Warp Shuffle):** ~3.5ms - 42.8x faster
- **Solution 6 (Grid-Stride):** ~2.8ms - **53.6x faster** 🏆

## Key Takeaways

1. **Memory access patterns matter more than algorithmic complexity.** The difference between solution 2 and 4 is just loop direction, but it doubles performance.

2. **Minimize synchronization.** Every `__syncthreads()` and atomic operation is a potential bottleneck. Warp-level operations avoid synchronization entirely.

3. **Do more work per thread.** Launching fewer blocks with more work per thread reduces overhead and contention, as long as you maintain occupancy.

## Testing Individual Solutions

Each solution is a standalone executable:

```bash
cd build/reduction
./01_reduction_atomic 100000000
./02_reduction_shared 100000000
./03_reduction_sequential 100000000
./04_reduction_contiguous 100000000
./05_warp_reduce 100000000
./06_grid_reduction 100000000
```

Each program outputs timing, speedup, and validation results.
