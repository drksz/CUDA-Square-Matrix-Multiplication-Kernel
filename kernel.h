#ifndef KERNEL_H
#define KERNEL_H


void matrixMult(float* matrix_M, float* matrix_N, float* matrix_P, int width, float *execTime);
__global__ void matrixMultKernel(float* M_copy, float* N_copy, float* P_copy, int width);


#endif