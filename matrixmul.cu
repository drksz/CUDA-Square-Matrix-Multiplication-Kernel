#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helpers.h"
#include "kernel.h"



int main()
{
    FILE* mat_M, *mat_N;

    mat_M = fopen("matrix_M.in", "r");
    mat_N = fopen("matrix_N.in", "r");

    if (mat_M == NULL || mat_N == NULL) {
        printf("\nFatal Error. No such file(s).\n\n");
        return 1;
    }

    // Initialize matrices M, N, and P with dimensions 1000 x 1000
    // Note that the matrices are in row-major order instead of
    // explicit multi-dimensional array declaration for easier memory
    // offset calculation and coalesced memory access pattern.
    float* matrix_M = (float*)malloc(1000*1000 * sizeof(float));
    float* matrix_N = (float*)malloc(1000*1000 * sizeof(float));
    float* matrix_P = (float*)malloc(1000*1000 * sizeof(float));

    if (!matrix_M || !matrix_N || !matrix_P) {
        printf("\nMemory allocation failed. Exiting program.\n\n");
        return 1;
    }

    scanInputMatrix(mat_M, mat_N, matrix_M, matrix_N, 1000);

    float execTime;

    //call to function matrixMult()
    matrixMult(matrix_M, matrix_N, matrix_P, 1000, &execTime);

    printMatrix(matrix_P, 1000);

    //prints multiplication kernel execution elapsed time
    printf("Matrix multiplication took %.6f seconds.\n\n", execTime/1000.0f);

    //frees dynamically allocated memory on the host side and closes the input files
    free(matrix_M);
    free(matrix_N);
    free(matrix_P);
    fclose(mat_M);
    fclose(mat_N);

    return 0;
}
