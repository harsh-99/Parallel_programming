__global__ void 
convolution(float *N , float *M, float *P , int Tile_Size, int Mask_Width , int Width)
{

int k = blockIdx.x * blockDim.x + threadIdx.x;
int l = blockIdx.y * blockDim.y + threadIdx.y;

__shared__ float N_ds[8][8];

int n = Mask_Width/2;
int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
int halo_index_top = (blockIdx.y - 1) * blockDim.y + threadIdx.y;
int halo_index_right = (blockIdx.x +1) * blockDim.x + threadIdx.x;
int halo_index_bottom = (blockIdx.y + 1) * blockDim.y + threadIdx.y;

N_ds[n + threadIdx.y][n + threadIdx.x]= N[(blockIdx.y * blockDim.y + threadIdx.y)*Width + (blockIdx.x * blockDim.x + threadIdx.x)];
if(threadIdx.x >= blockDim.x-n && threadIdx.y >= blockDim.y - n) {
    N_ds[threadIdx.y -(blockDim.y - n)][threadIdx.x -(blockDim.x - n)] = (halo_index_left < 0 || halo_index_top < 0)?0:N[halo_index_top* Width + halo_index_left];
    N_ds[threadIdx.y -(blockDim.y - n)][n+threadIdx.x] = (halo_index_top<0)?0:N[halo_index_top*Width + (blockDim.x*blockIdx.x + threadIdx.x)];
    N_ds[n+threadIdx.y][threadIdx.x -(blockDim.x - n)] = (halo_index_left<0)?0:N[(blockDim.y*blockIdx.y + threadIdx.y)*Width + halo_index_left];
}
if(threadIdx.x < n && threadIdx.y >= blockDim.y - n){ 
    N_ds[threadIdx.y -(blockDim.y - n)][n + blockDim.x + threadIdx.x] = (halo_index_right >= Width || halo_index_top < 0)?0:N[halo_index_top*Width + halo_index_right];
    N_ds[threadIdx.y -(blockDim.y - n)][n+threadIdx.x] = (halo_index_top<0)?0:N[halo_index_top*Width + (blockDim.x*blockIdx.x + threadIdx.x)];
    N_ds[n+threadIdx.y][n + blockDim.x + threadIdx.x] = (halo_index_right >= Width)?0:N[(blockDim.y*blockIdx.y + threadIdx.y)*Width + halo_index_right];
}
if(threadIdx.y < n && threadIdx.x >= blockDim.x - n){ 
    N_ds[n + blockDim.y + threadIdx.y][threadIdx.x -(blockDim.x - n)] = (halo_index_bottom >= Width || halo_index_left < 0)?0:N[halo_index_bottom*Width + halo_index_left];
    N_ds[n + blockDim.y + threadIdx.y][n+threadIdx.x] = (halo_index_bottom >= Width)?0:N[halo_index_bottom*Width + (blockDim.x*blockIdx.x + threadIdx.x)];
    N_ds[n+threadIdx.y][threadIdx.x -(blockDim.x - n)] = (halo_index_left < 0)?0:N[(blockDim.y*blockIdx.y + threadIdx.y)*Width + halo_index_left];
}
if(threadIdx.x < n && threadIdx.y < n){ 
    N_ds[n + blockDim.y + threadIdx.y][n + blockDim.x + threadIdx.x] = (halo_index_right >= Width || halo_index_bottom >= Width)?0:N[halo_index_bottom*Width + halo_index_right];
    N_ds[n + blockDim.y + threadIdx.y][n+threadIdx.x] = (halo_index_bottom >= Width)?0:N[halo_index_bottom*Width + (blockDim.x*blockIdx.x + threadIdx.x)];
    N_ds[n+threadIdx.y][n + blockDim.x + threadIdx.x] = (halo_index_right >=Width)?0:N[(blockDim.y*blockIdx.y + threadIdx.y)*Width + halo_index_right];
}
if(threadIdx.y < n && threadIdx.x > n && threadIdx.x <= blockDim.x-n) N_ds[n + blockDim.y + threadIdx.y][n + threadIdx.x] = (halo_index_bottom >=Width)?0:N[(halo_index_bottom*Width) + (blockIdx.x * blockDim.x + threadIdx.x)];
if(threadIdx.x < n && threadIdx.y > n && threadIdx.y <= blockDim.y-n) N_ds[n + threadIdx.y][n + blockDim.x + threadIdx.x] = (halo_index_right >=Width)?0:N[(blockDim.y * blockIdx.y + threadIdx.y)*Width + (halo_index_right)];
if(threadIdx.y >= blockDim.y - n && threadIdx.x > n && threadIdx.x <= blockDim.x-n) N_ds[threadIdx.y -(blockDim.y - n)][n+threadIdx.x] = (halo_index_top < 0)?0:N[(halo_index_top*Width) + (blockDim.x*blockIdx.x + threadIdx.x)];
if(threadIdx.x >= blockDim.x - n && threadIdx.y > n && threadIdx.y <= blockDim.y-n) N_ds[n+threadIdx.y][threadIdx.x -(blockDim.x - n)] = (halo_index_left < 0)?0:N[(blockDim.y * blockIdx.y + threadIdx.y)*Width + halo_index_left];


__syncthreads();

float Pvalue = 0;
for(int i =0; i < Mask_Width; i++){
    for(int j =0; j < Mask_Width ; j++) {
        Pvalue += N_ds[threadIdx.y + i][threadIdx.x + j] * M[(i*Mask_Width) + j];
    }
}
P[(l*Width) + k] = Pvalue;
}
