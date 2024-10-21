clean:
	rm mandel_c
	rm mandel2.bmp
	rm -r mandel_c.dSYM

main: mandel_c.c
	nvcc -o mandel_c mandel_c.c
	./mandel_c 1
