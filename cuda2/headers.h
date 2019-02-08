#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
__global__ void process_kernel1(const float*, const float*, float*, int);

__global__ void process_kernel2(float*, float*, int);
__global__ void process_kernel3(float*, float*, int);
