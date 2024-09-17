mandel_mpi: mandel_mpi.c
	mpicc -Wall -Werror -o2 -g3 mandel_mpi.c -o mandel_mpi

run: mandel_mpi
	mpirun -n 3 ./mandel_mpi 1

clear:
	rm mandel_mpi
	rm mandel_c
	rm *.bmp
	rm -r *.dSYM

speedup: mandel_mpi mandel_c
	time ./mandel_c 0
	time mpirun -n 3 ./mandel_mpi 0

mandel_c: mandel_c.c
	gcc -Wall -Werror -o2 -g3 mandel_c.c -o mandel_c