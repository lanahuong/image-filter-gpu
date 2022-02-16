#include <cuda.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <argp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "FreeImage.h"

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

const char *argp_program_version = "version 1.0";

enum filter {NONE, SAT, HMIR, VMIR, SBLUR, BLUR, GREY, SOBEL, POP, NEG, BIN};

struct arguments {
  char *inputFile;
  enum filter f;
  int filter_arg;
};

static int parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *a = (struct arguments*) state->input;
  switch (key) {
  case 's':
    a->f = SAT;
    if (!strcmp(arg, "r") || !strcmp(arg, "R")) {
      a->filter_arg = 0;
    } else if (!strcmp(arg, "g") || !strcmp(arg, "G")) {
      a->filter_arg = 1;
    } else if  (!strcmp(arg, "b") || !strcmp(arg, "B")) {
      a->filter_arg = 2;
    } else {
      argp_usage(state);
    }
    break;
  case 'f':
    a->f = SBLUR;
    a->filter_arg = (int) strtol(arg, NULL, 10);
    break;
  case 'b':
    a->f = BLUR;
    a->filter_arg = (int) strtol(arg, NULL, 10);
    break;
  case 'h':
    a->f = HMIR;
    break;
  case 'v':
    a->f = VMIR;
    break;
  case 'g':
    a->f = GREY;
    break;
  case 'e':
    a->f = SOBEL;
    break;
  case 'p':
    a->f = POP;
    break;
  case 'n':
    a->f = NEG;
    break;
  case 'i':
    a->f = BIN;
    a->filter_arg = 255 * (int) strtol(arg, NULL, 10) / 100;
  case ARGP_KEY_ARG:
    // If there are more than one argument show usage
    if (state->arg_num > 1) {
      argp_usage(state);
      return ARGP_ERR_UNKNOWN;
    }
    // Else store the argument as input file
    a->inputFile = arg;
    break;
  case ARGP_KEY_END:
    // If there are less than one argument show usage
    if (state->arg_num < 1)
      a->inputFile = "img.jpg";
    break;
  default:
    return ARGP_ERR_UNKNOWN;
    break;
  }
  return 0;
}

int main (int argc , char** argv)
{
  static struct argp_option options[] = {
      {"sat", 's', "RGB", 0, "Saturate componant R, G or B of the image"},
      {"static-blur", 'f', "n", 0, "Apply an average blur as describe in the subject n times"},
      {"blur", 'b', "r", 0, "Apply an average blur with radius r"},
      {"h-mirror", 'h', 0, 0, "Mirror the image horizontally"},
      {"v-mirror", 'v', 0, 0, "Mirror the image vertically"},
      {"grey", 'g', 0, 0, "Change the image to greyscale"},
      {"sobel", 'e', 0, 0, "Apply the the SOBEL filter for edge detection"},
      {"pop-art", 'p', 0, 0, "Apply the pop-art filter"},
      {"negative", 'n', 0, 0, "Negate the image"},
      {"binary", 'i', "percentage", 0, "Change the image to black and white"},
      {0}
};
  struct argp argp = {options, parse_opt, "FILE", 0};
  struct arguments arguments = {NULL, NONE, 0};
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  FreeImage_Initialise();
  //const char *PathName = "img.jpg";
  const char *PathName = arguments.inputFile;
  const char *PathDest = "new_img.jpg";
  // load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

  FIBITMAP* bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

  if(! bitmap )
    exit( 1 ); //WTF?! We can't even allocate images ? Die !

  unsigned width  = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch  = FreeImage_GetPitch(bitmap);

  fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

  unsigned image_size = width*height;
  unsigned alloc_size = sizeof(unsigned int) * image_size * 3;

  unsigned int *img;
  unsigned int *d_img, *d_img_tmp;

  // CPU allocation
  img = (unsigned int*) malloc(alloc_size);

  // GPU memory allocation
  cudaMalloc((void **) &d_img, alloc_size);
  if (arguments.f == HMIR || arguments.f == VMIR || arguments.f == SOBEL || arguments.f == SBLUR || arguments.f == BLUR || arguments.f == POP) {
    cudaMalloc((void **) &d_img_tmp, alloc_size);
  }

  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      int idx = ((y * width) + x) * 3;
      img[idx + 0] = pixel[FI_RGBA_RED];
      img[idx + 1] = pixel[FI_RGBA_GREEN];
      img[idx + 2] = pixel[FI_RGBA_BLUE];
      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  // Block and grid size
  dim3 blockSize(32,32);
  dim3 gridSize(0,0);
  gridSize.x = width / 32 +1;
  gridSize.y = height / 32 +1;

  // Timing treatments
  float duration = 0.;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Data transfer
  cudaMemcpy(d_img, img, alloc_size, cudaMemcpyHostToDevice);
  if (arguments.f == HMIR || arguments.f == VMIR || arguments.f == SOBEL || arguments.f == BLUR) {
    cudaMemcpy(d_img_tmp, img, alloc_size, cudaMemcpyHostToDevice);
  }

  // Kernel calls
  switch (arguments.f) {
    case SAT:
      cudaEventRecord(start);
      kernel_sat1<<<gridSize,blockSize>>>(d_img, (unsigned) arguments.filter_arg, image_size);
      break;
    case GREY:
      cudaEventRecord(start);
      kernel_grey<<<gridSize,blockSize>>>(d_img, image_size);
      break;
    case HMIR:
      cudaEventRecord(start);
      kernel_hmirror<<<gridSize,blockSize>>>(d_img, d_img_tmp, width, height);
      break;
    case VMIR:
      cudaEventRecord(start);
      kernel_vmirror<<<gridSize,blockSize>>>(d_img, d_img_tmp, width, height);
      break;
    case SBLUR:
      cudaEventRecord(start);
      for (int i = 0; i<arguments.filter_arg; i++) {
        cudaMemcpy(d_img_tmp, img, alloc_size, cudaMemcpyHostToDevice);
        kernel_blur<<<gridSize,blockSize>>>(d_img, d_img_tmp, width, height);
        cudaMemcpy(img, d_img, alloc_size, cudaMemcpyDeviceToHost);
      }
      break;
    case BLUR:
      cudaEventRecord(start);
      run_blur(img, d_img, d_img_tmp, width, height, blockSize, gridSize, arguments.filter_arg);
      //run_blur_(img, d_img, d_img_tmp, width, height, blockSize, gridSize);
      break;
    case SOBEL:
      cudaEventRecord(start);
      run_sobel(img, d_img, d_img_tmp, width, height, blockSize, gridSize);
      break;
    case POP:
      cudaEventRecord(start);
      run_popart(img, d_img, d_img_tmp, width, height);
      break;
    case NEG:
      cudaEventRecord(start);
      kernel_negative<<<gridSize,blockSize>>>(d_img, image_size);
      break;
    case BIN:
      cudaEventRecord(start);
      kernel_binary<<<gridSize,blockSize>>>(d_img, image_size, arguments.filter_arg);
      break;
  }

  // Copy back
  if (arguments.f != POP && arguments.f != SBLUR)
    cudaMemcpy(img, d_img, alloc_size, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);

  bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      RGBQUAD newcolor;

      int idx = ((y * width) + x) * 3;
      newcolor.rgbRed = img[idx + 0];
      newcolor.rgbGreen = img[idx + 1];
      newcolor.rgbBlue = img[idx + 2];

      if(!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
      { fprintf(stderr, "(%d, %d) Fail...\n", x, y); }

      pixel+=3;
    }
    // next line
    bits += pitch;
  }

  if( FreeImage_Save (FIF_JPEG, bitmap , PathDest , 0 ))
    cout << "Image successfully saved ! " << endl ;
  FreeImage_DeInitialise(); //Cleanup !

  cout << "Treatement time : " << duration << "ms" << endl ;

  free(img);
  cudaFree(d_img);
  cudaFree(d_img_tmp);
  //free(d_tmp);
}
