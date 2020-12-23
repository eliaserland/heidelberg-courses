#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>	// FFT
#include <pcg_basic.h>	// RNG

#define M_PI 3.14159265358979323846

#define DIM 2		// No. of dimensions
#define NGRID 256	// No. of gridpoints per dimension

/*
 * 	Fundamentals of Simulation Methods - WiSe 2020/2021
 * 	Problem Set 6 - Exercise 2
 *
 *   	Author: Elias Olofsson
 * 	Date: 2020-12-22
 */
 
// Data type for storing all information for a single particle.
typedef struct {
 	double pos[DIM];
 	double vel[DIM];
 	double acc[DIM];
 	double acc_prev[DIM];
 	double m;
} particle;


// Return a random double from the flat distribution on [0,1).
double rand_double(pcg32_random_t *rngptr) {
	return ldexp(pcg32_random_r(rngptr) ,-32);
}	


/** 
 * initialize() - Initialize the particles in the system.
 *  
 * @p: Pointer to array of particle structs.
 * @L: Length of box in one dimension.
 * @N: Number of particles.
 * @rngptr: Pointer to RNG.
 * 
 * Returns: Nothing.
 */
void initialize(particle *p, double L, int NP, pcg32_random_t *rngptr) {
	
	// Assign a random position within the box to each particle.
	for (int i = 0; i < NP; i++) {
		for (int d = 0; d < DIM; d++) {
			p[i].pos[d] = L*rand_double(rngptr);
			p[i].m = 1.0; // Unit mass.
		}
	}
	/*
	// Assign particels to a rectangular grid.
	for (int i = 0; i < NP; i++) {
		for (int d = 0; d < DIM; d++) {
			p[i].pos[d] = 0;
		}
	}*/	
	
	// Velocities and accelerations.
	for (int i = 0; i < NP; i++) {
		for (int d = 0; d < DIM; d++) {
			p[i].vel[d] = 0;
			p[i].acc[d] = 0;
			p[i].acc_prev[d] = 0;	
		}
	}	
}

/** 
 * print_pos() - Print particle positions to the terminal.
 * 
 * @p: Pointer to array of particle structs.
 * @NP: Number of particles
 *
 * Returns: Nothing.   
 */
void print_pos(particle *p, int NP) {
	
	for (int i = 0; i < NP; i++) {
		for (int d = 0; d < DIM; d++) {
			printf("%lf ", p[i].pos[d]);
		}
		printf("\n");	
	}
}


void calc_density(particle *p, double *density, int NP, double h) {
	
	// Volume of a single grid cell.
	const double v = pow(h, DIM);
	
	// Set the entire density field to zero.
	for (int i = 0; i < pow(NGRID, DIM); i++) {
		density[i] = 0;
	}
	
	// Dynamic allocations.
	double *W = malloc(pow(NGRID, DIM) * sizeof(double));
	double *pos_int = malloc(DIM * sizeof(int));	// LOOK OVER
	double *pos_star  = malloc(DIM*sizeof(double)) // LOOK OVER
	 
	// FLOATING POINT INDEX.
	 
	// For each particle:
	for (int i = 0; i < NP; i++) {
		/* Get the grid point closest to the particle, and the volume 
		   fraction of the grid cell in its lower corner. */
		for (int d = 0; d < DIM; d++) {
			pos_int[d] = floor(p[i].pos[d]);  // WRONG! REDO
			p_star[d] = pos_int[d] - p[i].pos[d]; // WRONG! REDO
		} 	
		
		for (int j = 0; j < pow(2,DIM); j++) {
		
		
		}
	}
	
	free(W);
	free(pos_int)
	free(pos_star);
	
}


int main(int argc, char **argv) {

	const int NP = 1;		// No. of particles to simulate.
	const double L = 1; 		// Box side length.
	
	
	
	const double h = L/NGRID; 	// Grid spacing. 
	
	
	// Declare and seed the RNG.
	pcg32_random_t rngptr;
	pcg32_srandom_r(&rngptr, time(NULL), (intptr_t)&rngptr);
	
	// Allocate storage for the particles.
	particle *p = malloc(NP * sizeof(particle));
 	
 	// Initialize the particles.
	initialize(p, L, NP, &rngptr);

	// Print position.
	print_pos(p, NP);


	// Allocate the density field.
	double *density = malloc(pow(NGRID,DIM)*sizeof(double));
	
	// Calulate the density field from the particle positions.
	calc_density() 
	 
	
		
	
	
	
	
	
	
		
	// Free dynamically allocated memory.
 	free(p);
 	free(density);
 	
 	return 0;
 }
