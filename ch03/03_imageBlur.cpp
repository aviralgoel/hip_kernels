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

unsigned char *load_image(const char *imagePath, int &width, int &height,
                          int &channels);
__global__ void HipImageBlur(unsigned char *image, unsigned char *blurredImage,
                             int width, int height, int patchSize);
int main() {
  // Path to the input image
  const char *imagePath = "../../resources/input/TF2.jpg";

  // Load the image
  int width, height, channels;
  unsigned char *img = load_image(imagePath, width, height, channels);

  int numOfPixels = width * height;

  unsigned char *d_blurredImage, *d_originalImage;
  HIP_CHECK(
      hipMalloc(&d_blurredImage, numOfPixels * 3 * sizeof(unsigned char)));
  HIP_CHECK(
      hipMalloc(&d_originalImage, numOfPixels * 3 * sizeof(unsigned char)));
  HIP_CHECK(hipMemcpy(d_originalImage, img,
                      numOfPixels * 3 * sizeof(unsigned char),
                      hipMemcpyHostToDevice));

  dim3 blockDim(32, 32);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);
  int patchSize = 5;

  hipLaunchKernelGGL(HipImageBlur, gridDim, blockDim, 0, 0, d_originalImage,
                     d_blurredImage, width, height, patchSize);

  unsigned char *h_blurredImage =
      (unsigned char *)malloc(numOfPixels * 3 * sizeof(unsigned char));
  HIP_CHECK(hipMemcpy(h_blurredImage, d_blurredImage,
                      numOfPixels * 3 * sizeof(unsigned char),
                      hipMemcpyDeviceToHost));

  // Save the grayscale image
  const char *output_image = "../../resources/output/output_blurred_GPU.jpg";
  if (stbi_write_jpg(output_image, width, height, 3, h_blurredImage, 90)) {
    std::cout << "Blurred image saved as " << output_image << std::endl;
  } else {
    std::cerr << "Failed to save blurred image." << std::endl;
  }

  stbi_image_free(img);

  free(h_blurredImage);
  HIP_CHECK(hipFree(d_blurredImage));
  HIP_CHECK(hipFree(d_originalImage));

  return 0;
}

unsigned char *load_image(const char *imagePath, int &width, int &height,
                          int &channels) {

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

__global__ void HipImageBlur(unsigned char *image, unsigned char *blurredImage,
                             int width, int height, int patchSize) {
  int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
  int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (pixel_x < width && pixel_y < height) {
    float pixelVal_r = 0.0f, pixelVal_g = 0.0f, pixelVal_b = 0.0f;
    int pixelCount = 0;

    for (int blurRow = -patchSize; blurRow <= patchSize; blurRow++) {
      int currentPixel_y = pixel_y + blurRow;

      if (currentPixel_y >= 0 && currentPixel_y < height) {
        for (int blurCol = -patchSize; blurCol <= patchSize; blurCol++) {
          int currentPixel_x = pixel_x + blurCol;

          if (currentPixel_x >= 0 && currentPixel_x < width) {
            int baseIndex = (currentPixel_y * width + currentPixel_x) * 3;
            pixelVal_r += image[baseIndex + 0];
            pixelVal_g += image[baseIndex + 1];
            pixelVal_b += image[baseIndex + 2];
            pixelCount++;
          }
        }
      }
    }

    // Normalize the accumulated values and write to the output image
    blurredImage[(pixel_y * width + pixel_x) * 3 + 0] =
        static_cast<unsigned char>(pixelVal_r / pixelCount);
    blurredImage[(pixel_y * width + pixel_x) * 3 + 1] =
        static_cast<unsigned char>(pixelVal_g / pixelCount);
    blurredImage[(pixel_y * width + pixel_x) * 3 + 2] =
        static_cast<unsigned char>(pixelVal_b / pixelCount);
  }
}
