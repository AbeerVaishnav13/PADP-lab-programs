#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int n_threads;

double** createMatrix(int N) {
	double **m = NULL;
	m = (double**) malloc(N * sizeof(double*));

	for (int i = 0; i < N; i++) {
		m[i] = (double*) malloc(N * sizeof(double));
	}
	return m;
}

void initializeMatrices(double **mat, double **L, double **U, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			mat[i][j] = i + j + 1;
			L[i][j] = 0;
			U[i][j] = 0;
		}
	}
}

void freeMatrix(double **m, int N) {
	for (int i = 0; i < N; i++) {
		free(m[i]);
	}
	free(m);
}

void LUdecomp(double **mat, double **L, double **U, int N) {
	int i, j, k;
	#pragma omp parallel for private(i, k) num_threads(n_threads)
	for(j = 0; j < N; j++) {
		for(i = 0; i < N; i++) {
			if(i <= j) {
				U[i][j] = mat[i][j];

				for(k = 0; k < i-1; k++)
					U[i][j] -= (L[i][k] * U[k][j]);

				if(i == j) L[i][j] = 1;
				else L[i][j] = 0;
			}
			else {
				L[i][j] = mat[i][j];
				for(k = 0; k <= j-1; k++)
					L[i][j] -= (L[i][k] * U[k][j]);

				L[i][j] /= U[j][j];
				U[i][j] = 0;
			}
		}
	}
}

int main() {
	int N;
	printf("Enter value of N: ");
	scanf("%d", &N);

	printf("Enter number of threads: ");
	scanf("%d", &n_threads);

	// Allocate memory to the matrices
	double **mat = createMatrix(N);
	double **L = createMatrix(N);
	double **U = createMatrix(N);

	// Initialize matrices
	initializeMatrices(mat, L, U, N);

	double t = omp_get_wtime();
	// Compute LU Decomposition
	LUdecomp(mat, L, U, N);
	t = omp_get_wtime() - t;

	char displayMat;
	getchar();
	printf("Do you want to display the LU matrix? (y/N): ");
	scanf("%c", &displayMat);

	if (displayMat == 'y' || displayMat == 'Y') {
		printf("Matrix mat:\n");
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				printf("%.2lf ", mat[i][j]);
			}
			printf("\n");
		}

		printf("Matrix L:\n");
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				printf("%.2lf ", L[i][j]);
			}
			printf("\n");
		}

		printf("Matrix U:\n");
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				printf("%.2lf ", U[i][j]);
			}
			printf("\n");
		}
	}

	// Free the matrix memory
	freeMatrix(mat, N);
	freeMatrix(L, N);
	freeMatrix(U, N);

	printf("Time taken: %lf", t);

	return 0;
}
