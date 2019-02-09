#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define tile_width 4
#define dimension 8 
#define mask_width 5
__global__ void convolution(float*, float*, float*, int, int, int);

