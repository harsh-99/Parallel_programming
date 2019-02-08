#include <headers.h>

int main(void){
	cudaError_t err = cudaSuccess;
	int datasize =  16384;
	size_t size = datasize*sizeof(float);
	float *h_a =(float *)malloc(size);
	float *h_b = (float *)malloc(size);
	float *h_c = (float *)malloc(size);
	float *h_d = (float *)malloc(size);
	float *h_e = (float *)malloc(size);
	
	if(h_a == NULL || h_b == NULL){
		fprintf(stderr, "Error in allocating host vectors\n");
		exit(EXIT_FAILURE);
	}
	for (int i=0; i<datasize ; i++){
		h_a[i] = rand()/(float)RAND_MAX;
		h_b[i] = rand()/(float)RAND_MAX;
	}
	float *d_a = NULL;
	err = cudaMalloc((void **)&d_a, size);
	if (err != cudaSuccess)
	    {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	    }
	float *d_b = NULL;
	err = cudaMalloc((void **)&d_b, size);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	float *d_c = NULL;
	err = cudaMalloc((void **)&d_c, size);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	float *d_d = NULL;
	err = cudaMalloc((void **)&d_d, size);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	float *d_e = NULL;
	err = cudaMalloc((void **)&d_e, size);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	
	dim3 X(4,2,2);
	dim3 Y(32,32,1);
	printf("Cuda first kernel launched\n");
	process_kernel1<<<X,Y>>>(d_a, d_b, d_c, datasize); 

	err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	dim3 X1(2,8,1);
	dim3 Y1(8,8,16);
	printf("Cuda kernel launched 2nd time\n");
	process_kernel2<<<X1,Y1>>>(d_c,d_d,datasize);

	dim3 X2(16,1,1);
	dim3 Y2(128,8,1);
	printf("Cuda kernel launched 3rd time\n");
	process_kernel3<<<X2,Y2>>>(d_d,d_e,datasize);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(h_d, d_d, size, cudaMemcpyDeviceToHost);	
	if (err != cudaSuccess)
	    {
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	    }
	err = cudaMemcpy(h_e, d_e, size, cudaMemcpyDeviceToHost);	
	if (err != cudaSuccess)
	    {
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	    }
    // Verify that the result vector is correct
    for (int i = 0; i < datasize; ++i)
    {
        if (fabs(sin(h_a[i]) + cos(h_b[i]) - h_c[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test1 PASSED\n");
	for (int i = 0; i < datasize; ++i)
    {
        if (fabs(log(h_c[i]) - h_d[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification for second failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test2 PASSED\n");
	for (int i = 0; i < datasize; ++i)
    {
        if (fabs(sqrt(h_d[i]) - h_e[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification for third failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test3 PASSED\n");

    // Free device global memory
    err = cudaFree(d_a);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_a);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_c);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaFree(d_d);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaFree(d_e);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d); 
    free(h_e);	

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
