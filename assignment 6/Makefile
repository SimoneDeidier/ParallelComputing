clean:
	rm mandel
	rm -r mandel.dSYM
	rm mandel1.bmp

main: mandel.cu
	nvcc -o mandel mandel.cu
	./mandel 1
