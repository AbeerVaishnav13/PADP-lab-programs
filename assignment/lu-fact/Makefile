CC1=clang
SRC1=sequentialLU.c
OUT1=cache/sequential

CC2=mpicc
SRC2=mpiLU.c
OUT2=cache/mpi

all: cache seq mpi

seq:
	$(CC1) -Xpreprocessor -fopenmp -lomp $(SRC1) -o $(OUT1)

mpi:
	$(CC2) $(SRC2) -o $(OUT2)

cache:
	mkdir cache

run:
	mpirun -np 3 $(OUT2)
