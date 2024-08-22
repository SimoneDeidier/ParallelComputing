#include <stdlib.h>
#include <stdio.h>
#include <time.h>	// used to initialize the random seed
#include <math.h>	// used for the abs() function
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

void copyAndAlterImg(uchar*, uchar*, int, int);

int main(void) {
	uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	readbmp("before.bmp", image);

	// Alter the image here

	// new image, double the size (3 uchar (1) * X * 2* Y * 2 => 12 * X * Y)
	uchar* newImg = calloc(XSIZE * YSIZE * 12, 1);
	// call to the function to copy the before image in the new one and alter it
	copyAndAlterImg(image, newImg, XSIZE, YSIZE);

	savebmp("after.bmp", newImg, XSIZE * 2, YSIZE * 2);

	free(image);
	free(newImg); // free all the heap memory allocated

	return 0;

}

/* procedure to double the image size and alter the colors
 *  oldImg = uchar vector of the before image
 *  newImg = uchar vector of the new image
 *  x = X dimension of the before image
 *  y = Y dimension of the before image
 */
void copyAndAlterImg(uchar* oldImg, uchar* newImg, int x, int y) {

	int iOffset = 0;
	int jOffset = 0;
	const int newX = x * 2;	// dimension of the new image (doubled on X and Y)
	const int newY = y * 2;
	int offset = 0;

	// initialize the random seed
	srand(time(NULL));

	// intialize the random offsets for the new colors of the new image
	int red_off = rand() % 256;
	int green_off = rand() % 256;
	int blue_off = rand() % 256;

	/* this cycle is to copy the before image in the new image
	 * copy the before image vector into the new one (allocated in main, passed by reference)
	 * the vector is seen as a matrix, each pixel (triplets of uchar) are copied in the new vector
	 * leaving a blank pixel after everyone and leaving a blank row between two copied rows
	 * complexity: O(N*M), where N and M are the X and Y dimensions of the old image
	 */
	for(int i = 0; i < y; i++) {
		for(int j = 0; j < x * 3; j += 3) {
			iOffset = x * i * 12;
			jOffset = 2 * j;
			newImg[iOffset + jOffset] = oldImg[(x * i * 3) + j];
			newImg[iOffset + jOffset + 1] = oldImg[(x * i * 3) + j + 1];
			newImg[iOffset + jOffset + 2] = oldImg[(x * i * 3) + j + 2];
		}
	}

	/* this cycle is for filling all the blank pixels in the copied rows
	 * the blank pixels are filled copying the previous one (the one of the before image copied)
	 * complexity: O(N'*M') where N' and M' are the X and Y dimensions of the new image
	 */
	for(int i = 0; i < newY; i++) {
		for(int j = 0; j < newX * 3; j += 3) {
			if(((j / 3) % 2) && !(i % 2)) {
				newImg[i * newX * 3 + j] = newImg[i * newX * 3 + j - 3];
				newImg[i * newX * 3 + j + 1] = newImg[i * newX * 3 + j - 2];
				newImg[i * newX * 3 + j + 2] = newImg[i * newX * 3 + j - 1];
			}
		}
	}

	/* this cycle is for filling all the blank rows of the new image
	 * the blank rows are filled with a opy of the previous row (the one made up with the copied
	 * pixels and the one filled with the last cycle)
	 * complexity: O(N'*M')
	 */
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

	/* this cycle changes the value of the R, G and B component of every pixel in the new image
	 * this is done subtracting the random offset (with absolute value to avoid overflow) to every uchar
	 * complexity: O(N'*M')
	 */
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