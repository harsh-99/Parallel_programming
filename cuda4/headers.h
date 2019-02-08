#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void convolution(float*, float*, float*, int, int, int);

