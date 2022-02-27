# Image filtering on GPU

This is a GPU programming project in CUDA.

## Requirements

You will of couse need an NVIDIA GPU and cuda installed.

You also need the FreeImage library. It may be available from repos in your distribution if not or if you are running on a cluster you can download it and install it from https://freeimage.sourceforge.io/download.html . I made this project with version 3.18.0.

## Building

If you are on a cluster environnement check the `env.sh` and change the modules for the ones available on your cluster. Then you need to source this file.

```sh
source env.sh
```

Then just run `make`.

## Filtering images

The executable is called `modif_img.exe`. It can filter JPEG images.

The flag `--help` will list the available filters. You can apply only one at a time on the image of your choice. The resulting image will be saved as `new_img.jpeg` in the current directory.

```sh
./modif_img.exe --help
Usage: modif_img.exe [OPTION...] FILE

  -b, --blur=r               Apply an average blur with radius r
  -e, --sobel                Apply the the SOBEL filter for edge detection
  -f, --static-blur=n        Apply an average blur using only direct neighbors
  -g, --grey                 Change the image to greyscale
  -h, --h-mirror             Mirror the image horizontally
  -i, --binary=percentage    Change the image to black and white
  -n, --negative             Negate the image
  -p, --pop-art              Apply the pop-art filter
  -s, --sat=RGB              Saturate componant R, G or B of the image
  -v, --v-mirror             Mirror the image vertically
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
  ```
