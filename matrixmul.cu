#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>



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


void printMatrix(float* matrix, int width) {

    if (!matrix) {
        printf("\nMatrix not found.\n");
        return;
    }

    printf("\nPrinting matrix...\n\n");

    for (int i = 0; i < width*width; i++) {
        if (i != 0 && i%width == 0) {
            printf("\n");
        }
        printf("%.1f ", matrix[i]);
    }

    printf("\n\nPrinting finished.\n\n");

}



int main()
{
    FILE* mat_M, *mat_N;

    mat_M = fopen("matrix_M.in", "r");
    mat_N = fopen("matrix_N.in", "r");

    if (mat_M == NULL || mat_N == NULL) {
        printf("\nFatal Error. No such file(s).\n\n");
        return 1;
    }

    float* matrix_M = (float*)malloc(1000*1000 * sizeof(float));
    float* matrix_N = (float*)malloc(1000*1000 * sizeof(float));
    float* matrix_P = (float*)malloc(1000*1000 * sizeof(float));

    if (!matrix_M || !matrix_N || !matrix_P) {
        printf("\nMemory allocation failed. Exiting program.\n\n");
        return 1;
    }

    int i = 0;
    while (i < (1000*1000) && fscanf(mat_M, "%f", &matrix_M[i]) && 
            fscanf(mat_N, "%f", &matrix_N[i])) 
        {i++;}

    
    float execTime;

    matrixMult(matrix_M, matrix_N, matrix_P, 1000, &execTime);

    printMatrix(matrix_P, 50);

    printf("Matrix multiplication took %.6f seconds.\n\n", execTime);

    free(matrix_M);
    free(matrix_N);
    free(matrix_P);
    fclose(mat_M);
    fclose(mat_N);

    return 0;
}