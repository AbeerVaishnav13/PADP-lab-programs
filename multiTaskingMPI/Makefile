CC=mpicc
SRC=multiTaskingMPI.c
OUT=multiTaskingMPI

all:
	$(CC) $(SRC) -o $(OUT)

run:
	mpirun -np 3 $(OUT)
