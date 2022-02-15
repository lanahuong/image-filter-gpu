#ifndef KERNELS_H_
#define KERNELS_H_

// Kernels
__global__ void kernel_sat1(unsigned int *img, int rgb_sat, unsigned size);
__global__ void kernel_grey(unsigned int *img, unsigned size);
__global__ void kernel_grey(unsigned int *img, unsigned size);
__global__ void kernel_hmirror(unsigned int *new_img, unsigned int *img,
                               unsigned width, unsigned height);
__global__ void kernel_blur(unsigned int *new_img, unsigned int *img,
                            unsigned width, unsigned height);
__global__ void kernel_convolution_rgb(unsigned int *new_img, unsigned int *img,
                                       unsigned width, unsigned height,
                                       float *kernel, unsigned int kernel_size);
__global__ void kernel_sobel(unsigned int *new_img, unsigned int *img,
                             unsigned width, unsigned height, int *kernels);
__global__ void kernel_reduction(unsigned int *new_img, unsigned int *img,
                                 unsigned width, unsigned height);

// Helper function for more complex opérations with kernels
void run_blur_(unsigned int *img, unsigned int *d_img, unsigned int *d_img_tmp,
               unsigned width, unsigned height, dim3 blockSize, dim3 gridSize);
void run_blur(unsigned int *img, unsigned int *d_img, unsigned int *d_img_tmp,
              unsigned width, unsigned height, dim3 blockSize, dim3 gridSize,
              int r);
void run_sobel(unsigned int *img, unsigned int *d_img, unsigned int *d_img_tmp,
               unsigned width, unsigned height, dim3 blockSize, dim3 gridSize);
void run_popart(unsigned int *img, unsigned int *d_img, unsigned width,
                unsigned height);

#endif // KERNELS_H_
