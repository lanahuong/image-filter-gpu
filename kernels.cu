#include <cuda.h>

__global__
void kernel_sat1(unsigned int* img, int rgb_sat, unsigned size) {
    // Compute index of thread
    int g_block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int th_block = blockDim.x * blockDim.y;
    int g_idx = g_block_idx * th_block + threadIdx.x + threadIdx.y * blockDim.x;

    // Compute index of pixel to alter
    int k = g_idx * 3;

    // Correct rgb_sat to be in {0,1,2}
    rgb_sat = rgb_sat % 3;

    // Saturate the componant rgb_sat
    if (g_idx<size) {
        img[k + rgb_sat] = 0xFF;
    }
}

__global__
void kernel_grey(unsigned int* img, unsigned size) {
    // Compute index of thread
    int g_block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int th_block = blockDim.x * blockDim.y;
    int g_idx = g_block_idx * th_block + threadIdx.x + threadIdx.y * blockDim.x;

    // Compute index of pixel to alter
    int k = g_idx * 3;

    // Compute grey color and assign it to all componant
    if (g_idx<size) {
        int grey = img[k+0]*0.299 + img[k+1]*0.587 + img[k+2]*0.114;
        img[k] = grey;
        img[k + 1] = grey;
        img[k + 2] = grey;
    }
}

__global__
void kernel_hmirror(unsigned int* new_img, unsigned int* img, unsigned width, unsigned height) {
    unsigned size = width * height;

    // Compute index of thread
    int g_block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int th_block = blockDim.x * blockDim.y;
    int g_idx = g_block_idx * th_block + threadIdx.x + threadIdx.y * blockDim.x;

    // Compute index of pixel to alter
    int k = g_idx * 3;

    // Compute the index of the last pixel in the line
    int i = ((g_idx / width) + 1) * width * 3 - 3;
    // Compute the column of the pixel to alter
    int j = (g_idx % width) * 3;

    // Assign each pixel the color of the opposite pixel on the line
    if (g_idx<size) {
        new_img[k] = img[i-j];
        new_img[k + 1] = img[i-j+1];
        new_img[k + 2] = img[i-j+2];
    }
}

__global__
void kernel_blur(unsigned int* new_img, unsigned int* img, unsigned width, unsigned height) {
    unsigned size = width * height;

    // Compute index of thread
    int g_block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int th_block = blockDim.x * blockDim.y;
    int g_idx = g_block_idx * th_block + threadIdx.x + threadIdx.y * blockDim.x;

    // Compute index of pixel to alter
    int k = g_idx * 3;

    // Compute the row and column of the pixel to alter
    int i = g_idx / width;
    int j = g_idx % width;

    if (g_idx<size && j > 0 && j < width - 1 && i > 0 && i < height - 1) {
        unsigned v[4] = {
                         ((i-1)*width+j)*3,
                         (i*width+j-1)*3,
                         (i*width+j+1)*3,
                         ((i+1)*width+j)*3,
                        };
        new_img[k] = (img[k] + img[v[0]] + img[v[1]] + img[v[2]] + img[v[3]]) / 5;
        new_img[k + 1] = (img[k + 1] + img[v[0] + 1] + img[v[1] + 1] + img[v[2] + 1] + img[v[3] + 1]) / 5;
        new_img[k + 2] = (img[k + 2] + img[v[0] + 2] + img[v[1] + 2] + img[v[2] + 2] + img[v[3] + 2]) / 5;
    }
}

__global__
void kernel_convolution_rgb(unsigned int* new_img, unsigned int* img, unsigned width, unsigned height, float* kernel, unsigned int kernel_size) {
    unsigned size = width * height;

    // Compute index of thread
    int g_block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int th_block = blockDim.x * blockDim.y;
    int b_idx = threadIdx.x + threadIdx.y * blockDim.x;
    int g_idx = g_block_idx * th_block + b_idx;

    // Compute index of pixel to alter
    int k = g_idx * 3;

    // Compute the row and column of the pixel to alter
    int i = g_idx / width;
    int j = g_idx % width;

    int kernel_center = (kernel_size - 1) / 2;
    // Efficiently copy the kernel in the block shared memory
    int ks2 = kernel_size*kernel_size;
    extern __shared__ float ker[];
    if (b_idx<ks2)
        ker[b_idx] = kernel[b_idx];
    __syncthreads();

    // Compute convolution for each color
    if (g_idx<size && j >= kernel_center && j < (width - kernel_center) && i >= kernel_center && i < (height - kernel_center)) {
        int r = 0;
        int g = 0;
        int b = 0;
        for (int ki = 0; ki<kernel_size; ki++) {
            for (int kj = 0; kj<kernel_size; kj++) {
                int ii = i + ki - kernel_center;
                int jj = j + kj - kernel_center;
                int kk = (ii*width + jj)*3;
                r += img[kk] * ker[ki * kernel_size + kj];
                g += img[kk + 1] * ker[ki * kernel_size + kj];
                b += img[kk + 2] * ker[ki * kernel_size + kj];
            }
        }
        new_img[k] = r;
        new_img[k + 1] = g;
        new_img[k + 2] = b;
    }
}

void run_blur_(unsigned int* img, unsigned int* d_img, unsigned int* d_img_tmp, unsigned width, unsigned height, dim3 blockSize, dim3 gridSize) {
  // Create the kernel and send it to the GPU
  float kernel[9] = {0.f, 0.2f, 0.f, 0.2f, 0.2f, 0.2f, 0.f, 0.2f, 0.f};
  float *d_kernel;
  cudaMalloc((void **) &d_kernel, 9*sizeof(float));
  cudaMemcpy(d_kernel, &kernel, 9*sizeof(float), cudaMemcpyHostToDevice);

  // Lauch the kernel
  kernel_convolution_rgb<<<gridSize,blockSize, 9*sizeof(float)>>>(d_img, d_img_tmp, width, height, d_kernel, 3);

  cudaFree(d_kernel);
}

void run_blur(unsigned int* img, unsigned int* d_img, unsigned int* d_img_tmp, unsigned width, unsigned height, dim3 blockSize, dim3 gridSize, int r) {
  int k_size = 2 * r + 1;
  int k_size2 = k_size * k_size;
  int k_alloc = sizeof(float) * k_size2;
  // Create the kernel and send it to the GPU
  float *kernel = (float*) malloc(k_alloc);
  float val = 1.f/(float)k_size2;
  for (int i = 0; i < k_size2; i++) {
      kernel[i] = val;
  }
  float *d_kernel;
  cudaMalloc((void **) &d_kernel, k_alloc);
  cudaMemcpy(d_kernel, kernel, k_alloc, cudaMemcpyHostToDevice);

  // Lauch the kernel
  kernel_convolution_rgb<<<gridSize,blockSize, k_alloc>>>(d_img, d_img_tmp, width, height, d_kernel, k_size);

  free(kernel);
  cudaFree(d_kernel);
}

__global__
void kernel_sobel(unsigned int* new_img, unsigned int* img, unsigned width, unsigned height, int* kernels) {
    unsigned size = width * height;

    // Compute index of thread
    int g_block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int th_block = blockDim.x * blockDim.y;
    int b_idx = threadIdx.x + threadIdx.y * blockDim.x;
    int g_idx = g_block_idx * th_block + b_idx;

    // Compute index of pixel to alter
    int k = g_idx * 3;

    // Compute the row and column of the pixel to alter
    int i = g_idx / width;
    int j = g_idx % width;

    // Efficiently copy the kernel in the block shared memory
    __shared__ float ker[9];
    if (b_idx<9)
        ker[b_idx] = kernels[b_idx];
    __syncthreads();

    if (g_idx<size && j >= 1 && j < (width - 1) && i >= 1 && i < (height - 1)) {
        int gx = 0;
        int gy = 0;
        for (int ki = 0; ki<3; ki++) {
            for (int kj = 0; kj<3; kj++) {
                int ii = i + ki - 1;
                int jj = j + kj - 1;
                int kk = (ii*width + jj)*3;
                gx += img[kk] * ker[ki * 3 + kj];
                gy += img[kk] * ker[kj * 3 + ki];
            }
        }
        int val = sqrt((float) (gx*gx + gy*gy));
        new_img[k] = val;
        new_img[k + 1] = val;
        new_img[k + 2] = val;
    }
}

void run_sobel(unsigned int* img, unsigned int* d_img, unsigned int* d_img_tmp, unsigned width, unsigned height, dim3 blockSize, dim3 gridSize) {
  unsigned image_size = width*height;
  unsigned alloc_size = sizeof(unsigned int) * image_size * 3;

  kernel_grey<<<gridSize,blockSize>>>(d_img, image_size);
  cudaMemcpy(img, d_img, alloc_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_img_tmp, img, alloc_size, cudaMemcpyHostToDevice);

  // Create the kernel and send it to the GPU
  int kernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  //float kernel[9] = {0.f, 0.2f, 0.f, 0.2f, 0.2f, 0.2f, 0.f, 0.2f, 0.f};
  int *d_kernel;
  cudaMalloc((void **) &d_kernel, 9*sizeof(int));
  cudaMemcpy(d_kernel, &kernel, 9*sizeof(int), cudaMemcpyHostToDevice);

  // Lauch the kernel
  kernel_sobel<<<gridSize,blockSize>>>(d_img, d_img_tmp, width, height, d_kernel);

  cudaFree(d_kernel);
}

__global__
void kernel_reduction(unsigned int* new_img, unsigned int* img, unsigned width, unsigned height) {
    unsigned size = width * height * 0.25;

    // Compute index of thread
    int g_block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int th_block = blockDim.x * blockDim.y;
    int g_idx = g_block_idx * th_block + threadIdx.x + threadIdx.y * blockDim.x;

    // Compute index of pixel to alter
    int k = g_idx * 3;

    // Compute the row and column of the pixel of the new image to set
    int i = 2 * g_idx / width;
    int j = g_idx % (width/2);

    if (g_idx<size) {
        // For each pixel we compute the average of a square of 4
        int ii = 2 * i;
        int jj = 2 * j;

        int kk = (ii * width + jj) * 3;
        int kk_next = ((ii+1) * width + jj) * 3;

        new_img[k] = (img[kk] + img[kk+3] + img[kk_next] + img[kk_next+3]) / 4;
        new_img[k + 1] = (img[kk + 1] + img[kk + 4] + img[kk_next + 1] + img[kk_next + 4]) / 4 ;
        new_img[k + 2] = (img[kk + 2] + img[kk + 5] + img[kk_next + 2] + img[kk_next + 5]) / 4;
    }
}

void run_popart(unsigned int* img, unsigned int* d_img, unsigned width, unsigned height) {
  unsigned image_size = width * height;
  unsigned alloc_size = sizeof(unsigned int) * image_size * 3;

  unsigned image_size_small = width * height / 4;
  unsigned alloc_size_small = sizeof(unsigned int) * image_size_small * 3;

  dim3 blockSize(32,32);
  dim3 gridSize(0,0);
  gridSize.x = width / 64 +1;
  gridSize.y = height / 64 +1;

  unsigned int *img_small = (unsigned int*) malloc(alloc_size_small);
  unsigned int* img_tmp = (unsigned int*) malloc(alloc_size);
  unsigned int *d_img_small;

  cudaMalloc((void **) &d_img_small, alloc_size_small);

  kernel_reduction<<<gridSize,blockSize>>>(d_img_small, d_img, width, height);

  cudaMemcpy(img_small, d_img_small, alloc_size_small, cudaMemcpyDeviceToHost);

  unsigned int nstreams = 4;
  cudaStream_t stream[nstreams];
  for (int i = 0; i<nstreams; i++) {
      cudaStreamCreate(&stream[i]);
  }

  for (int i = 0; i<nstreams; i++) {
    int offset = i * image_size_small * 3;
    cudaMemcpyAsync(d_img+offset, img_small, alloc_size_small, cudaMemcpyHostToDevice, stream[i]);
    if (i == 0) {
      kernel_grey<<<gridSize,blockSize,0,stream[i]>>>(d_img, image_size_small);
    } else {
        kernel_sat1<<<gridSize,blockSize,0,stream[i]>>>(d_img+offset, i, image_size_small);
    }
    cudaMemcpyAsync(img_tmp+offset, d_img+offset, alloc_size_small, cudaMemcpyDeviceToHost, stream[i]);
  }

  cudaDeviceSynchronize();

  int k = 0;
  int kk = 0;
  for (int i = 0; i< height/2; i++) {
      for (int j = 0; j<3*width/2; j++) {
          img[k] = img_tmp[kk];
          k++;
          kk++;
      }
      kk += 3*image_size_small - 3*width/2;
      for (int j = 0; j<3*width/2; j++) {
          img[k] = img_tmp[kk];
          k++;
          kk++;
      }
      kk -= 3*image_size_small;
  }
  kk += 3*image_size_small;
  for (int i = 0; i< height/2; i++) {
      for (int j = 0; j<3*width/2; j++) {
          img[k] = img_tmp[kk];
          k++;
          kk++;
      }
      kk += 3*image_size_small - 3*width/2;
      for (int j = 0; j<3*width/2; j++) {
          img[k] = img_tmp[kk];
          k++;
          kk++;
      }
      kk -= 3*image_size_small;
  }

  for (int i = 0; i<nstreams; i++) {
      cudaStreamDestroy(stream[i]);
  }

  free(img_small);
  cudaFree(d_img_small);
}
