#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#define SEED 35791246

int main() {
	int n = 10000, i, count = 0;
	double z, pi, x,y;

	srand(SEED);

	#pragma omp parallel for private(x, y, z) reduction(+: count) num_threads(4)
	for (i = 0; i < n; i++) {
		x = (double)rand()/RAND_MAX;
		y = (double)rand()/RAND_MAX;
		z = x*x+y*y;
		if (z<=1) count++;
	}
	pi = (double)count/n*4;

	printf("# of trials= %d , estimate of pi is %g \n",n,pi);

	return 0;
}
