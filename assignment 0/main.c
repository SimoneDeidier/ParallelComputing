#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

void copyAndAlterImg(uchar*, uchar*, int, int);

int main(void) {
	uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	readbmp("before.bmp", image);

	// Alter the image here

	// new image, double the size (3 uchar * X * 2* Y * 2 => 12 * X * Y)
	uchar* newImg = calloc(XSIZE * YSIZE * 12, 1);
	copyAndAlterImg(image, newImg, XSIZE, YSIZE);

	savebmp("after.bmp", newImg, XSIZE * 2, YSIZE * 2);

	free(image);
	free(newImg);

	return 0;

}

void copyAndAlterImg(uchar* oldImg, uchar* newImg, int x, int y) {

	int iOffset = 0;
	int jOffset = 0;
	const int newX = x * 2;
	const int newY = y * 2;
	int offset = 0;

	srand(time(NULL));

	int red_off = rand() % 256;
	int green_off = rand() % 256;
	int blue_off = rand() % 256;

	for(int i = 0; i < y; i++) {
		for(int j = 0; j < x * 3; j += 3) {
			iOffset = x * i * 12;
			jOffset = 2 * j;
			newImg[iOffset + jOffset] = oldImg[(x * i * 3) + j];
			newImg[iOffset + jOffset + 1] = oldImg[(x * i * 3) + j + 1];
			newImg[iOffset + jOffset + 2] = oldImg[(x * i * 3) + j + 2];
		}
	}

	for(int i = 0; i < newY; i++) {
		for(int j = 0; j < newX * 3; j += 3) {
			if(((j / 3) % 2) && !(i % 2)) {
				newImg[i * newX * 3 + j] = newImg[i * newX * 3 + j - 3];
				newImg[i * newX * 3 + j + 1] = newImg[i * newX * 3 + j - 2];
				newImg[i * newX * 3 + j + 2] = newImg[i * newX * 3 + j - 1];
			}
		}
	}

	for(int i = 0; i < newY; i++) {
		if(i % 2) {
			for(int j = 0; j < newX * 3; j += 3) {
				offset = i * newX * 3 + j;
				newImg[offset] = newImg[offset - newX * 3];
				newImg[offset + 1] = newImg[offset - newX * 3 + 1];
				newImg[offset + 2] = newImg[offset - newX * 3 + 2];
			}
		}
	}

	for(int i = 0; i < newX * newY * 3; i++) {
		switch(i % 3) {
			case 0: {
				newImg[i] = abs(newImg[i] - red_off);
				break;
			}
			case 1: {
				newImg[i] = abs(newImg[i] - blue_off);
				break;
			}
			case 2: {
				newImg[i] = abs(newImg[i] - green_off);
				break;
			}
			default: {
				exit(1);
			}
		}
	}

	return;

}