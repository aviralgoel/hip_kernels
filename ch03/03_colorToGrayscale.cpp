#include <hip/hip_runtime.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stbi/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stbi/stb_image_write.h"

#define HIP_CHECK(expression)                                                  \
  {                                                                            \
    const hipError_t status = expression;                                      \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      exit(-1);                                                                \
    }                                                                          \
  }

unsigned char *load_image(const char *imagePath, int &width, int &height);
__global__ void HipColorToGrayScaleSingleChannel(unsigned char *img,
                                                 unsigned char *grayImg,
                                                 int width, int height);
void convertOnCPU(unsigned char *img, int width, int height);

int main() {
  // Path to the input image
  const char *imagePath = "../../resources/input/TF2.jpg";

  // Load the image
  int width, height;
  unsigned char *img = load_image(imagePath, width, height);
  if (!img)
    return -1; // Exit if image loading fails

  int numOfPixels = width * height;

  // Device memory for the input image and the single-channel output
  unsigned char *d_img, *d_grayImg;
  HIP_CHECK(hipMalloc(&d_img, numOfPixels * 3 *
                                  sizeof(unsigned char))); // original image
  HIP_CHECK(hipMalloc(
      &d_grayImg, numOfPixels * sizeof(unsigned char))); // future gray image
  HIP_CHECK(hipMemcpy(d_img, img, numOfPixels * 3 * sizeof(unsigned char),
                      hipMemcpyHostToDevice));

  // Define kernel launch dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);

  hipLaunchKernelGGL(HipColorToGrayScaleSingleChannel, gridDim, blockDim, 0, 0,
                     d_img, d_grayImg, width, height);

  // Allocate host memory for the grayscale image and copy the result back
  unsigned char *grayImage =
      (unsigned char *)malloc(numOfPixels * sizeof(unsigned char));
  HIP_CHECK(hipMemcpy(grayImage, d_grayImg, numOfPixels * sizeof(unsigned char),
                      hipMemcpyDeviceToHost));

  // Save the grayscale image
  const char *output_image = "../../resources/output/output_grayscale_GPU.jpg";
  if (stbi_write_jpg(output_image, width, height, 1, grayImage, 90)) {
    std::cout << "Grayscale image saved as " << output_image << std::endl;
  } else {
    std::cerr << "Failed to save grayscale image." << std::endl;
  }

  // Convert on CPU
  // convertOnCPU(img, width, height);

  // Clean up
  stbi_image_free(img);
  HIP_CHECK(hipFree(d_img));
  HIP_CHECK(hipFree(d_grayImg));
  free(grayImage);

  return 0;
}

void convertOnCPU(unsigned char *img, int width, int height) {

  unsigned char *grayscaleImg = new unsigned char[width * height];
  int channels = 3; // RGB
  for (int i = 0; i < width * height; i++) {
    int r = img[i * channels + 0]; // Red
    int g = img[i * channels + 1]; // Green
    int b = img[i * channels + 2]; // Blue

    // Weighted grayscale conversion
    grayscaleImg[i] = static_cast<unsigned char>(0.3 * r + 0.59 * g + 0.11 * b);
  }

  // Save the grayscale image
  const char *outputPath = "output_grayscale_CPU.jpg";
  if (stbi_write_jpg(outputPath, width, height, 1, grayscaleImg, 90)) {
    std::cout << "Grayscale image saved as " << outputPath << std::endl;
  } else {
    std::cerr << "Failed to save grayscale image." << std::endl;
  }

  delete[] grayscaleImg;
}

unsigned char *load_image(const char *imagePath, int &width, int &height) {
  int channels;
  unsigned char *img = stbi_load(imagePath, &width, &height, &channels, 0);
  if (!img) {
    std::cerr << "Failed to load image: " << imagePath << std::endl;
    return nullptr;
  }
  if (channels != 3) {
    std::cerr << "This is not a 3-channel image." << std::endl;
    stbi_image_free(img);
    return nullptr;
  }

  std::cout << "Image file loaded: " << width << "x" << height << " with "
            << channels << " channels.\n";
  return img;
}

__global__ void HipColorToGrayScaleSingleChannel(unsigned char *img,
                                                 unsigned char *grayImg,
                                                 int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int pixelIndex = (y * width + x);
    int pixelInRowMajor = pixelIndex * 3;

    float r = img[pixelInRowMajor + 0];
    float g = img[pixelInRowMajor + 1];
    float b = img[pixelInRowMajor + 2];

    unsigned char gray =
        static_cast<unsigned char>(0.3f * r + 0.59f * g + 0.11f * b);
    grayImg[pixelIndex] = gray;
  }
}
