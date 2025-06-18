#include <cuda_runtime.h>
#include "kernel.h"

void matrixMult(float* matrix_M, float* matrix_N, float* matrix_P, int width, float *execTime) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int size = width*width * sizeof(float);

    float* M_copy, *N_copy, *P_copy;

    cudaMalloc((void**)&M_copy, size);
    cudaMalloc((void**)&N_copy, size);
    cudaMalloc((void**)&P_copy, size);

    cudaMemcpy(M_copy, matrix_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_copy, matrix_N, size, cudaMemcpyHostToDevice);    
    
    dim3 blockDimension(16,16, 1);
    dim3 gridDimension;

    gridDimension.x = (width + blockDimension.x - 1) / blockDimension.x;
    gridDimension.y = (width + blockDimension.y - 1) / blockDimension.y;
    gridDimension.z = 1;

    cudaEventRecord(start, 0);


    //kernel invocation here
    matrixMultKernel<<<gridDimension, blockDimension>>>(M_copy, N_copy, P_copy, width);


    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(execTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(matrix_P, P_copy, size, cudaMemcpyDeviceToHost);

    cudaFree(M_copy);
    cudaFree(N_copy);
    cudaFree(P_copy);

}

__global__ void matrixMultKernel(float* M_copy, float* N_copy, float* P_copy, int width) {

    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    float dotProduct = 0;


    if (rowIdx < width && colIdx < width) {
     
        for (int k = 0; k < width; k++) {
            dotProduct += M_copy[rowIdx*width + k] * N_copy[k*width + colIdx];
        }

        P_copy[rowIdx*width + colIdx] = dotProduct;

    }

}