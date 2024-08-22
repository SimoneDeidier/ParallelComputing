main: main.o bitmap.o
	gcc -Wall -Werror -o2 -g3 main.o bitmap.o -o main

main.o: main.c
	gcc -c main.c

bitmap.o: bitmap.c
	gcc -c bitmap.c

clear:
	rm *.o
	rm main
	rm after.bmp