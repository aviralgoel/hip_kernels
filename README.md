### Table of Content

- [Table of Content](#table-of-content)
- [Hello World Kernel ch01](#hello-world-kernel-ch01)
- [Vector Addition Kernel ch02](#vector-addition-kernel-ch02)
- [Color to Greyscale Conversion Kernel ch03](#color-to-greyscale-conversion-kernel-ch03)
### <a name="hello-world-kernel"></a>Hello World Kernel ch01
**Description**: Simple kernel that prints hello world from GPU threads along with their thread IDs and block IDs

### <a name="vector addition"></a>Vector Addition Kernel ch02
**Description**: 1D kernel that performs addition of two vectors. Each thread reads a single element value from vector and vector B and performs addition to store a single element in vector C.

### <a name="color-to-greyscale-conversion-kernel"></a>Color to Greyscale Conversion Kernel ch03
**Description**: A kernel with 2D grid of 2D blocks of 16x16 threads. Each thread then converts the tuple of rgb pixel values into single float value on a grayscale spectrum.

