#include "headers.h"
/**
 * Host main routine
 */

float check(float *a, float *b){
    float c;
    c = a[0*dimension+1]*b[1*mask_width+0] + a[0*dimension+2]*b[1*mask_width+1] + a[0*dimension+3]*b[1*mask_width+2] + a[1*dimension+1]*b[2*mask_width+0]+ a[1*dimension+2]*b[2*mask_width+1] + a[1*dimension+3]*b[2*mask_width+2]; 
    return c;
}

int main(void)
{
    cudaError_t err = cudaSuccess;
    // int tile_width = 4;
    // int dimension = 8;
    int numElements = dimension*dimension;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A = (float *)malloc(size);

    if (h_A == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    float *h_C = (float *)malloc(size);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // int mask_width = 5;
    int num_mask = mask_width*mask_width;
    size_t size1 = num_mask * sizeof(float);

    float *h_B = (float *)malloc(size1);

    if (h_B == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        // h_B[i] = rand()/(float)RAND_MAX;
    }
    for (int i = 0; i < num_mask; ++i)
    {
        h_B[i] = rand()/(float)RAND_MAX;
        // h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i=0; i<numElements; i++){
        printf("%f   ", h_A[i]); 
        if ((i+1)%dimension == 0){
            printf("\n");
        }
    }

    for (int i=0; i<num_mask; i++){
        printf("%f   ", h_B[i]); 
        if ((i+1)%mask_width == 0){
            printf("\n");
        }
    }
    
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_B, h_B, size1, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int a = (dimension/tile_width);
    dim3 X1(a,a,1);
    dim3 Y1(tile_width,tile_width,1);
    printf("Cuda kernel launched\n");
    convolution<<<X1,Y1>>>(d_A, d_B, d_C, tile_width, mask_width, dimension);


    // int threadsPerBlock = 1024;
    // int blocksPerGrid =1;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    // matrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements, dimension);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i=0; i<numElements; i++){
        printf("%f  ",h_C[i]);
        if ((i+1)%dimension == 0){
            printf("\n");
        }
    }

    printf("Checking one element\n");
    float check_value;
    check_value = check(h_A, h_B); 
    printf("The value checked is %f\n", check_value); 	
    // Verify that the result vector is correct
    // for (int i = 0; i < numElements; ++i)
    // {
    //     if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    //     {
    //         fprintf(stderr, "Result verification failed at element %d!\n", i);
    //         exit(EXIT_FAILURE);
    //     }
    // }

    // printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

