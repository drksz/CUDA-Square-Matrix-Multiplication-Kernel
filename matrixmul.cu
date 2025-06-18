#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>



__global__ void matrix_mult(float* matrix_M, float* matrix_N, float* matrix_P, int width) {

    int size = width*width * sizeof(float);

    

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

    float* matrix_M = (float*)malloc(50*50 * sizeof(float));
    float* matrix_N = (float*)malloc(50*50 * sizeof(float));
    float* matrix_P = (float*)malloc(50*50 * sizeof(float))

    if (!matrix_M || !matrix_N) {
        printf("\nMemory allocation failed. Exiting program.\n\n");
        return 1;
    }

    int i = 0;
    while (i < 2500 && fscanf(mat_M, "%d", &matrix_M[i]) && 
            fscanf(mat_N, "%d", &matrix_N[i])) 
        {i++;}

    
    /*
    *
    *
    *   rest of code here
    * 
    * 
    */



    free(matrix_M);
    free(matrix_N);
    free(matrix_P);
    fclose(mat_M);
    fclose(mat_N);

    return 0;
}