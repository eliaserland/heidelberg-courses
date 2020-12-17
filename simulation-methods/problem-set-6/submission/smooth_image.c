#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#define M_PI 3.14159265358979323846
#define PIXELS  512

/*
 * 	Fundamentals of Simulation Methods - WiSe 2020/2021
 * 	Problem Set 6 - Exercise 1
 *
 *   	Author: Elias Olofsson
 * 	Date: 2020-12-16
 */

/* 
 * Reads a square image in 8-bit/color PPM format from the given file. 
 * Note: No checks on valid format are done. 
 */
void read_image(char *fname, int pixels, double *red, double *green, double *blue)
{
	FILE *fd;

  	if ((fd = fopen(fname, "r"))) {
      		int row, col, width, height, maxvalue;

     		fscanf(fd, "P6 %d %d %d ", &width, &height, &maxvalue);

      		for(row=0; row < pixels; row++) {
			for(col=0; col < pixels; col++) {
	    			unsigned char rgb[3];
	    			fread(rgb, 3, sizeof(char), fd);

		    		red[row*pixels + col] 	= rgb[0];
				green[row*pixels + col] = rgb[1];
				blue[row*pixels + col]  = rgb[2];
	  		}
	  	}
     		fclose(fd);
    	} else {
      		printf("file %s not found\n", fname);
      		exit(1);
    	}
}

/*
 * Writes a square image in 8-bit/color PPM format.
 */
void write_image(char *fname, int pixels, double *red, double *green, double *blue)
{
	FILE *fd;

  	if ((fd = fopen(fname, "w"))) {
      		int row, col;

      		fprintf(fd, "P6\n%d %d\n%d\n", pixels, pixels, 255);

      		for(row=0; row < pixels; row++) {
			for(col=0; col < pixels; col++) {
		    		unsigned char rgb[3];
		    		rgb[0] = red[row*pixels + col];
		    		rgb[1] = green[row*pixels + col];
		    		rgb[2] = blue[row*pixels + col];

	    			fwrite(rgb, 3, sizeof(char), fd);
	  		}
	  	}
      		fclose(fd);
    	} else {
      		printf("file %s can't be opened\n", fname);
      		exit(1);
    	}
}

/* Sum the values for each color channel and print the results. */
void sum_rgb(double *red, double *green, double *blue) {
	
	double sum;
	double *color;
	char str[10];

	for (int coloridx = 0; coloridx < 3; coloridx++) {
		if (coloridx == 0) {
			color = red;
			strcpy(str, "Red"); 
		} else if (coloridx == 1) {
			color = green;
			strcpy(str, "Green");
		} else {
			color = blue;
			strcpy(str, "Blue");
		}
		
		sum = 0;
		for (int i = 0; i < PIXELS*PIXELS; i++) {
			sum += color[i];
		}
		printf("%-6s: %.2lf\n", str, sum);
	}
	printf("\n");
}


int main(int argc, char **argv)
{
	int i, j, colindex;
	double *red, *green, *blue, *color;
  
	/* allocate some storage for the image, and then read it */

	red   = malloc(PIXELS * PIXELS * sizeof(double));
	green = malloc(PIXELS * PIXELS * sizeof(double));
  	blue  = malloc(PIXELS * PIXELS * sizeof(double));

  	read_image("aq-original.ppm", PIXELS, red, green, blue);

	printf("\nSum of pixel values per color channel, original image.\n");
	sum_rgb(red, green, blue);

  	/* Now we set up our desired smoothing kernel. We'll use complex number for it even though it is real. */

  	/* first, some space allocation */
  	fftw_complex *kernel_real = fftw_malloc(PIXELS * PIXELS * sizeof(fftw_complex));
  	fftw_complex *kernel_kspace = fftw_malloc(PIXELS * PIXELS * sizeof(fftw_complex));

	double r_h, ix, jx;
  	double hsml = 10.0;
	double k = 40./(7.*M_PI*hsml*hsml); 

  	/* now set the values of the kernel */
  	for (i=0; i < PIXELS; i++) {
    		for (j=0; j < PIXELS; j++) {
			kernel_real[i*PIXELS + j][0] = 0;  /* real part */
			kernel_real[i*PIXELS + j][1] = 0;  /* imaginary part */

			/* do something sensible here to set the real part of the kernel */
				
			if (i >= PIXELS/2) {
				ix = i-PIXELS;
			} else {
				ix = i;
			}
			if (j >= PIXELS/2) {
				jx = j-PIXELS;
			} else {
				jx = j;
			}
			
			r_h = sqrt(ix*ix + jx*jx)/hsml;
			
			if (r_h < 0.5) {
				kernel_real[i*PIXELS + j][0] = k*(1.+6.*r_h*r_h*(r_h-1.));
			} else if (r_h < 1.0) {
				kernel_real[i*PIXELS + j][0] = k*2.*(1.-r_h)*(1.-r_h)*(1.-r_h);
			}
      		}
	}
  

  	/* Let's calculate the Fourier transform of the kernel */
  	/* the FFTW3 library used here requires first a 'plan' creation, followed by the actual execution */
  
  	fftw_plan plan_kernel = fftw_plan_dft_2d (PIXELS, PIXELS, kernel_real, kernel_kspace, FFTW_FORWARD, FFTW_ESTIMATE);

  	/* now do the actual transform */
  	fftw_execute(plan_kernel);


  	/* further space allocations for image transforms */
  	fftw_complex *color_real   = fftw_malloc(PIXELS * PIXELS * sizeof(fftw_complex));
  	fftw_complex *color_kspace = fftw_malloc(PIXELS * PIXELS * sizeof(fftw_complex));

  	/* create corresponding FFT plans */
  	fftw_plan plan_forward  = fftw_plan_dft_2d (PIXELS, PIXELS, color_real, color_kspace, FFTW_FORWARD, FFTW_ESTIMATE);
  	fftw_plan plan_backward = fftw_plan_dft_2d (PIXELS, PIXELS, color_kspace, color_real, FFTW_BACKWARD, FFTW_ESTIMATE);

  
	/* we now convolve each color channel with the kernel using FFTs */
	for (colindex = 0; colindex < 3; colindex++) {
      		if (colindex == 0) {
			color = red;
      		} else if (colindex == 1) {
			color = green;
      		} else {
			color = blue;
		}

      		/* copy input color into complex array */
      		for (i=0; i < PIXELS; i++) {
			for (j=0; j < PIXELS; j++) {
	    			color_real[i*PIXELS + j][0] = color[i*PIXELS + j];    /* real part */
	    			color_real[i*PIXELS + j][1] = 0;                      /* imaginary part */
	  		}
	  	}
	 
      		/* forward transform */
      		fftw_execute(plan_forward);

      		/* multiply with kernel in Fourier space */
      		for (i=0; i < PIXELS; i++) {
			for (j=0; j < PIXELS; j++) {
		    		color_kspace[i*PIXELS + j][0] *= kernel_kspace[i*PIXELS + j][0]; 
		     		color_kspace[i*PIXELS + j][1] *= kernel_kspace[i*PIXELS + j][1]; 
	  		}
  		}

      		/* backward transform */
      		fftw_execute(plan_backward);
            
      		/* copy real value of complex result back into color array */
      		for(i=0; i < PIXELS; i++) {
			for(j=0; j < PIXELS; j++) {
	  			color[i*PIXELS + j] = color_real[i*PIXELS + j][0] / (PIXELS * PIXELS);
	  		}
  		}
    	}
      
  	write_image("aq-smoothed.ppm", PIXELS, red, green, blue);

	printf("Sum of pixel values per color channel, smoothed image.\n");
	sum_rgb(red, green, blue);
  	
	/* Clean up all dynamically allocated memory. */
	fftw_destroy_plan(plan_kernel);
	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);

  	fftw_free(kernel_real);  	
  	fftw_free(kernel_kspace);
  	fftw_free(color_real);
  	fftw_free(color_kspace);
  	
  	fftw_cleanup();

	free(red);
  	free(green);
  	free(blue);

  	return 0;
}


