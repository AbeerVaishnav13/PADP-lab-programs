CC=clang
MPICC=mpicc
SEQ=lu_seq.c
OMP=lu_openmp.c
MPI=lu_mpi.c

all: seq omp mpi_my

seq:
	$(CC) $(SEQ) -o cache/seq

omp:
	$(CC) $(OMP) -o cache/omp -Xpreprocessor -fopenmp -lomp

mpi_my:
	$(MPICC) $(MPI) -o cache/mpi

run:
	./cache/seq 1000
	./cache/omp 1000 4
	mpirun -np 4 ./cache/mpi 1000
