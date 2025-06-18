#include <stdio.h>
#include "helpers.h"

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

void scanInputMatrix(FILE* mat_src_A, FILE* mat_src_B, float* matrix_A, float* matrix_B, int width) {

    int i = 0;
    while (i < (width*width) && fscanf(mat_src_A, "%f", &matrix_A[i]) && 
            fscanf(mat_src_B, "%f", &matrix_B[i])) 
        {i++;}

}