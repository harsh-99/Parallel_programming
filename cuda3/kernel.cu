__global__ void
matrix(float *A, int numElements, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    float a; 
    if (i < n && j< n && i!= (n-1) && i%2==0)
    {
        	a = A[j*n + i];
        	A[j*n + i] = A[j*n + i + 1];
        	A[j*n + i +1] = a;	 
    }
    if (i<n&& j<n && i<j){
        A[i*n+j] = A[j*n + i];
    }	
    
}

