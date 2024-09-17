#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define XSIZE 2560
#define YSIZE 2048

#define MAXITER 255

double xleft=-2.01;
double xright=1;
double yupper,ylower;
double ycenter=1e-6;
double step;

int pixel[XSIZE*YSIZE];

#define PIXEL(i,j) ((i)+(j)*XSIZE)

typedef struct {
	double real,imag;
} complex_t;

/* function to calculate the values of the mandelbrot set
 * this function is parametric with the number of the processes that runs
 * rank: rank of the process that is running the function
 * nProc: total number of processes that are running
 */
void calculate(int rank, int nProc) {

    // indices to divide in chunks of row for each process
    int start = (YSIZE / nProc) * rank;
    /* if i'm the last process i have to fill also the last row
     * (case where the image dim cannot be divided by the number of proc.)
     */
    int end = rank != (nProc - 1) ? (YSIZE / nProc) * (rank + 1) : YSIZE;

    // vector to save locally the values calculated by the process
    int data[XSIZE * YSIZE];

	for(int i = 0; i < XSIZE; i++) {
        // rows index goes from start to end calculated before
		for(int j = start; j < end; j++) {

            // same code as sequential

			/* Calculate the number of iterations until divergence for each pixel.
			   If divergence never happens, return MAXITER */
			complex_t c,z,temp;
			int iter=0;
			c.real = (xleft + step*i);
			c.imag = (ylower + step*j);
			z = c;
			while(z.real*z.real + z.imag*z.imag < 4) {
				temp.real = z.real*z.real - z.imag*z.imag + c.real;
				temp.imag = 2*z.real*z.imag + c.imag;
				z = temp;
				if(++iter==MAXITER) break;
			}
			data[PIXEL(i,j)]=iter;
		}
	}
    // all the processes except the one with rank = 0 have to send the data
    if(rank) {
        MPI_Send(data, XSIZE*YSIZE, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else {
        /* unite all the data from all the processes into one array
         * iterate on the number of processes
         */
        for(int p = 0; p < nProc; p++) {
            // receive data from other processes before gathering them
            if(p) {
                MPI_Recv(data, XSIZE*YSIZE, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // calculate the indexes (same as the start)
            start = (YSIZE / nProc) * p;
            end = p != (nProc - 1) ? (YSIZE / nProc) * (p + 1) : YSIZE;
            // copy the data calculated by the processes in the pixel array
            for(int i = 0; i < XSIZE; i++) {
                for(int j = start; j < end; j++) {
                    pixel[PIXEL(i, j)] = data[PIXEL(i, j)];
                }
            }
        }
    }
}

typedef unsigned char uchar;

/* save 24-bits bmp file, buffer must be in bmp format: upside-down */
void savebmp(char *name,uchar *buffer,int x,int y) {
	FILE *f=fopen(name,"wb");
	if(!f) {
		printf("Error writing image to disk.\n");
		return;
	}
	unsigned int size=x*y*3+54;
	uchar header[54]={'B','M',size&255,(size>>8)&255,(size>>16)&255,size>>24,0,
		0,0,0,54,0,0,0,40,0,0,0,x&255,x>>8,0,0,y&255,y>>8,0,0,1,0,24,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	fwrite(header,1,54,f);
	fwrite(buffer,1,XSIZE*YSIZE*3,f);
	fclose(f);
}

/* given iteration number, set a colour */
void fancycolour(uchar *p,int iter) {
	if(iter==MAXITER);
	else if(iter<8) { p[0]=128+iter*16; p[1]=p[2]=0; }
	else if(iter<24) { p[0]=255; p[1]=p[2]=(iter-8)*16; }
	else if(iter<160) { p[0]=p[1]=255-(iter-24)*2; p[2]=255; }
	else { p[0]=p[1]=(iter-160)*2; p[2]=255-(iter-160)*2; }
}

int main(int argc,char **argv) {
	if(argc==1) {
		puts("Usage: MANDEL n");
		puts("n decides whether image should be written to disk (1=yes, 0=no)");
		return 0;
	}

	/* Calculate the range in the y-axis such that we preserve the
	   aspect ratio */
	step=(xright-xleft)/XSIZE;
	yupper=ycenter+(step*YSIZE)/2;
	ylower=ycenter-(step*YSIZE)/2;

    int comm_sz, my_rank;   // variables to store the # of proc. and rank

    /* initialize MPI
     * set the communicator size as the processes passed as arguments with mpirun
     * set the rank for each process
     */
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // each process will run the calculate function
	calculate(my_rank, comm_sz);

    // only the first process (rank = 0) will do I/O operations (save the img)
    if(!my_rank) {
        // same code as the sequential one
        if(strtol(argv[1],NULL,10)!=0) {
            /* create nice image from iteration counts. take care to create it upside
            down (bmp format) */
            unsigned char *buffer=calloc(XSIZE*YSIZE*3,1);
            for(int i=0;i<XSIZE;i++) {
                for(int j=0;j<YSIZE;j++) {
                    int p=((YSIZE-j-1)*XSIZE+i)*3;
                    fancycolour(buffer+p,pixel[PIXEL(i,j)]);
                }
            }
            /* write image to disk */
            savebmp("mandel2.bmp",buffer,XSIZE,YSIZE);
        }
    }

    // finalize MPI
    MPI_Finalize();

	return 0;
}
