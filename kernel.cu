#include <cuda_runtime.h>
#include "kernel.h"

/*
    This function is the starting point of where the computation task is handed over to the GPU.
    The process is divided into three parts:
    
    Part 1: memory is allocated onto the device (GPU) to hold copies of input data (the matrices). 
    
    Part 2: kernel is invoked and ran on multiple threads to parallelize the matrix multiplcation computation. 
    
    Part 3: Results are then copied from the device back to the host (CPU) and previously allocated 
        device memory is freed.
*/
void matrixMult(float* matrix_M, float* matrix_N, float* matrix_P, int width, float *execTime) {

    cudaEvent_t start, stop;    //cudaEvent_t objects to track kernel execution time
    
    //creates the start and stop events
    cudaEventCreate(&start);    
    cudaEventCreate(&stop);     

    int size = width*width * sizeof(float);     //size for all matrix copies 

    float* M_copy, *N_copy, *P_copy; 

    //allocating memory for matrix copies onto the device
    cudaMalloc((void**)&M_copy, size);
    cudaMalloc((void**)&N_copy, size);
    cudaMalloc((void**)&P_copy, size);


    //the contents of matrices M and N are copied onto the allocated memory
    cudaMemcpy(M_copy, matrix_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_copy, matrix_N, size, cudaMemcpyHostToDevice);    
    

    /*Block and grid dimension initialization*/

    dim3 blockDimension(16,16, 1);      //blocks are set to have 16x16 threads each
    dim3 gridDimension;

    
        //To calculate all elements of the product matrix, we need a sufficient amount of 
        //threads for each element. 

        //We want  " gridDimension.x * blockDimension >= width "

        //To get gridDimension, simply divide width by the specified block dimension.
        //The addition of the block dimension and subtraction of 1 in the statement to simulate
        //rounding up in integer division and also ensures that there are enough blocks to be launched. 

    
    gridDimension.x = (width + blockDimension.x - 1) / blockDimension.x;
    gridDimension.y = (width + blockDimension.y - 1) / blockDimension.y;
    gridDimension.z = 1;


    cudaEventRecord(start, 0);      //starts the time tracking


    //Matrix multiplication kernel is invoked here
    //kernel is launched on multiple threads defined by 
    //the dimensions set earlier
    matrixMultKernel<<<gridDimension, blockDimension>>>(M_copy, N_copy, P_copy, width);



    cudaEventRecord(stop,0);        //ends the time tracking

    cudaEventSynchronize(stop);     //simply waits for the GPU/device to finish all work

    cudaEventElapsedTime(execTime, start, stop);    //function to calculate time taken for kernel to execute


    //frees resources allocated for the start and stop events
    //similar to what you do with malloc()
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    //copies the result of the multiplication to the original matrix from the GPU
    cudaMemcpy(matrix_P, P_copy, size, cudaMemcpyDeviceToHost);


    //frees device allocated memory
    cudaFree(M_copy);
    cudaFree(N_copy);
    cudaFree(P_copy);

}



/*
    This is the multiplication kernel that is run on the threads. The "__global__" keyword 
    means that this function is called from the CPU/host and run on the GPU/device.
*/
__global__ void matrixMultKernel(float* M_copy, float* N_copy, float* P_copy, int width) {

    //calculates the row and column index for a product matrix element,
    //row of matrix M, and col of matrix N to perform a dot product on
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    
    float dotProduct = 0;       //variable to accumulate the dot product result on


    //since the width of the matrices are not divisible by the block dimension, an 
    //if-statement is used for bounds checking to prevent out-of-bounds memory access
    if (rowIdx < width && colIdx < width) {
     
        //iterates over the elements of the row of matrix M 
        //and col of matrix N simultaneously for dot product calculation
        for (int k = 0; k < width; k++) {
            dotProduct += M_copy[rowIdx*width + k] * N_copy[k*width + colIdx];
        }


        P_copy[rowIdx*width + colIdx] = dotProduct;   //dot product is assigned to product matrix
                                                      //at (rowIdx, colIdx)

    }

}