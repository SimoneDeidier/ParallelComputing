clean:
	rm mandel
	rm -r mandel.dSYM

main: mandel.cu
	nvcc -o mandel mandel.cu
	./mandel 1
