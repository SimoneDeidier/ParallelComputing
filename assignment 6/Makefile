clean:
	rm mandel
	rm mandel2.bmp
	rm -r mandel.dSYM

main: mandel.cu
	nvcc -o mandel mandel.cu
	./mandel 1
