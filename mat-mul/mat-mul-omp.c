#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
	int **arr1, **arr2, **arr3;
	int size = 1000;
	int thread_count = 4;
	int i, j, k;

	// Init
	arr1 = (int**) malloc(size * sizeof(int*));
	for(i = 0; i < size; i++)
		arr1[i] = (int*) malloc(size * sizeof(int));

	arr2 = (int**) malloc(size * sizeof(int*));
	for(i = 0; i < size; i++)
		arr2[i] = (int*) malloc(size * sizeof(int));

	arr3 = (int**) malloc(size * sizeof(int*));
	for(i = 0; i < size; i++)
		arr3[i] = (int*) malloc(size * sizeof(int));

	// Assigning values
	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++)
			arr1[i][j] = (i+1)*(j+1);
	}

	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++)
			arr2[i][j] = (i+1)*(j+1);
	}

	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++)
			arr3[i][j] = 0;
	}

	#pragma omp parallel for private(j, k) num_threads(thread_count)
	// Multiplication
	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			int sum = 0;
			for(k = 0;  k < size; k++) {
				sum += arr1[i][k] * arr2[k][j];
			}
			arr3[i][j] = sum;
		}
	}
}
