### Table of Content

- [Table of Content](#table-of-content)
- [Hello World Kernel ch01](#hello-world-kernel-ch01)
- [Vector Addition Kernel ch02](#vector-addition-kernel-ch02)
- [Color to Greyscale Conversion Kernel ch03](#color-to-greyscale-conversion-kernel-ch03)
- [Image Blur Kernel ch03](#image-blur-kernel-ch03)
- [Matrix Matrix Multiplication Kernel ch03](#matrix-matrix-multiplication-kernel-ch03)

### <a name="hello-world-kernel-ch01"></a>[Hello World Kernel ch01](./ch01/01_helloWorld.cpp)
**Description**: Simple kernel that prints hello world from GPU threads along with their thread IDs and block IDs

### <a name="vector-addition-kernel-ch02"></a>[Vector Addition Kernel ch02](./ch02/02_vecAdd.cpp)
**Description**: 1D kernel that performs addition of two vectors. Each thread reads a single element value from vector and vector B and performs addition to store a single element in vector C.

### <a name="color-to-greyscale-conversion-kernel-ch03"></a>[Color to Greyscale Conversion Kernel ch03](./ch03/03_colorToGrayscale.cpp)
**Description**: A kernel with 2D grid of 2D blocks of 16x16 threads. Each thread then converts the tuple of rgb pixel values into single float value on a grayscale spectrum.

### <a name="image-blur-kernel-ch03"></a>[Image Blur Kernel ch03](./ch03/03_imageBlur.cpp)
**Description**: A kernel with 2D grid of 2D blocks of 32x32 threads. Each thread then sums the tuple of rgb pixel values into a cumulative average of neighbouring pixel values (patch size).

### <a name="square-matrix-multiplication-ch03"></a>[Matrix Matrix Multiplication Kernel ch03](./ch03/03_matrixMultiplication.cpp)
**Description**: A 2D grid with 2D block of 32x32 threads kernel that multiplies two square matrices and verifies the result with CPU implementation.

### Build Instructions

To build and run the kernels:

1. Create a build directory and navigate to it:
    ```bash
    mkdir -p build
    cd build
    ```

2. Run CMake and make to build a specific target:
    ```bash
    cmake ..
    make <target_name>
    ```

3. Execute the compiled binary:
    ```bash
    ./<chapter_directory>/<binary_name>
    ```

Example:
```bash
cmake .. && make 03_colorToGrayscale && ./ch03/03_colorToGrayscale
```

This will build and run the color to grayscale conversion kernel, which processes the input image and saves the result to `../resources/output/output_grayscale_GPU.jpg`.
