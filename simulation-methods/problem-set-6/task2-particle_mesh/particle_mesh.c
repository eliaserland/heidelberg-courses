#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>	// Fast Fourier Transform from http://www.fftw.org/ 
#include <pcg_basic.h>	// Random Number Generator from https://www.pcg-random.org/

#define M_PI 3.14159265358979323846

#define DIM 2		// No. of dimensions (ONLY WORKING IN 2D FOR NOW)
#define NGRID 256	// No. of gridpoints per dimension

/*
 * 	Fundamentals of Simulation Methods - WiSe 2020/2021
 * 	Problem Set 6 - Exercise 2
 *
 *   	Author: Elias Olofsson
 * 	Date: 2020-12-22
 */
 
/* Data type for storing all information for a single particle. */
typedef struct {
 	double pos[DIM];
 	double vel[DIM];
 	double acc[DIM];
 	double acc_prev[DIM];
 	double m;
} particle;


/* Return a random double from the flat distribution on [0,1). */
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

/** 
 * fprint_density() - Print particle density field to a file.
 * 
 * @density: Pointer to the density field.
 *
 * Returns: Nothing.   
 */
void fprint_density(fftw_complex *density) {
	
	// Create an output file.
	char fname[100];
	sprintf(fname, "output_density.txt");
	FILE *fd = fopen(fname, "w");
	
	for (int i = 0; i < pow(NGRID,DIM); i++) {
		fprintf(fd, "%lf ", density[i][0]);	// Print the real part.
		if (i%NGRID == NGRID-1) {
			fprintf(fd, "\n");
		}		
	}
}

/**
 * calc_density() - Update the density field based on particle positions.
 *
 * @p: 	Pointer to array of particles.
 * @density: 	Pointer to the density field.
 * @NP: 	Number of particles.
 * @h: 	One dimensional grid spacing.
 *
 * Returns: Nothing.
 */
void calc_density(particle *p, fftw_complex *density, int NP, double h) {
	
	// Volume of a single grid cell.
	const double v = pow(h, DIM);
	
	// Set the entire density field to zero.
	for (int i = 0; i < pow(NGRID, DIM); i++) {
		density[i][0] = 0; // Real part.
		density[i][1] = 0; // Imaginary part.
	}
	
	// Dynamic allocations.
	double *W = malloc(pow(2,DIM) * sizeof(double));  // Volume fraction
	double *idx_float = malloc(DIM * sizeof(double)); // Floating point index
	double *p_star    = malloc(DIM * sizeof(double)); // Length fraction 
	
	int idx_int[DIM]; 	// Lower corner index.
	int idx[DIM];		// Current gridpoint index.
	int lin_idx;		// Linear index of current gridpoint.
	
	// For each particle:
	for (int i = 0; i < NP; i++) {		
		// For each dimension:
		for (int d = 0; d < DIM; d++) {
			// Calculate  floating point index of the particle.
			idx_float[d] = p[i].pos[d]/h - 0.5;  

			// Index of the gridpoint to lower side of the particle.
			idx_int[d] = (int)floor(idx_float[d]);
			
			// Length fraction belonging to the gridpoint on the lower side.
			p_star[d] = idx_float[d] - idx_int[d];
		} 	
		
		// Set volume fractions to unity.
		for (int j = 0; j < pow(2,DIM); j++) {
			W[j] = 1; 
		}
		
		// For each of the two closest gridpoints in each dimension:
		for (int j = 0; j < pow(2,DIM); j++) {
			
			/* Find the Cloud-in-Cell volume fraction W for the particle
			   at the gridpoint. */
			for (int d = 0; d < DIM; d++) {
				if (((int)floor(j/pow(2,d)))%2 == 0) {
					W[j] *= 1.0 - p_star[d];
				} else {
					W[j] *= p_star[d];
				}
			}
			
			/* Get the linear index of the gridpoint corresponding to
			   the density field. */
			lin_idx = 0;
			for (int d = 0; d < DIM; d++) {
				// Index of the gridpoint in the current dimension.
				idx[d] = idx_int[d] + ((int)floor(j/pow(2,d)))%2;
			
				// Periodically map back points outside the grid.
				if (idx[d] == -1) {
					idx[d] += NGRID;
				}
				if (idx[d] == NGRID) {
					idx[d] -= NGRID;
				}
				
				/* Add contribution from current dimension to the
				   linear index. */
				lin_idx += idx[d] * pow(NGRID,d);
			}

			/* Assign to the density field, the mass fraction for the
			   particle at the current gridpoint. */			
			density[lin_idx][0] = p[i].m / v * W[j];  	// Real part.
			density[lin_idx][1] = 0; 		 	// Imaginary part.
		}
	}
	// Free dynamically allocated memory.
	free(W);
	free(idx_float);
	free(p_star);
}


int main(int argc, char **argv) {

	/* Simulation settings */
	const int NP = 1;		// No. of particles to simulate.
	const double L = 1; 		// Box side length.
	
	// -------------------------------------------------
	const double h = L/NGRID; 	// Grid spacing. 
	
	// Declare and seed the RNG used for the particle initialization.
	pcg32_random_t rngptr;
	pcg32_srandom_r(&rngptr, time(NULL), (intptr_t)&rngptr);
	
	// Allocate storage for the particles.
	particle *p = malloc(NP * sizeof(particle));
 	
 	// Initialize the particles.
	initialize(p, L, NP, &rngptr);

	// Print the positions of the particles. (DEBUGGING)
	print_pos(p, NP);

	// Allocations.
	fftw_complex *density_real   = fftw_malloc(pow(NGRID,DIM) * sizeof(fftw_complex));
	fftw_complex *density_kspace = fftw_malloc(pow(NGRID,DIM) * sizeof(fftw_complex));
	fftw_complex *greens_kspace  = fftw_malloc(pow(NGRID,DIM) * sizeof(fftw_complex));
	fftw_complex *pot_real	      = fftw_malloc(pow(NGRID,DIM) * sizeof(fftw_complex));

	// Create FFTW plans for forward and backwards transformations.
	fftw_plan plan_forward = fftw_plan_dft_2d(NGRID, NGRID, density_real, density_kspace,
	 					   FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan plan_backward = fftw_plan_dft_2d (NGRID, NGRID, density_kspace, pot_real,
						     FFTW_BACKWARD, FFTW_ESTIMATE);
	
	int idx[DIM]; 
	double k2, l2;
	
	// Set up the Green's function in Fourier space.
	for (int i = 0; i < pow(NGRID,DIM); i++) {
		
		// Convert linear index i to DIM-dimensional indices.
		for (int d = 0; d < DIM; d++) {
			idx[d] = i % (int)pow(NGRID,d);
		
			// Convert to fourier frequency indices.
			if (idx[d] >= NGRID/2) {
				idx[d] -= NGRID;
			}
		}
		
		// Determine the squared sum of the Fourier indices.
		l2 = 0;
		for (int d = 0; d < DIM; d++) {
			l2 += idx[d]*idx[d];
		}
		
		// Squared sum of the k-vector.
		k2 = 4.0*M_PI*M_PI/(L*L)*l2;		
		
		// Set the real part of the Green's function.
		if (k2 == 0) {
			greens_kspace[i][0] = 0; // Set zero-frequency to 0.
		} else {
			greens_kspace[i][0] = -4.0*M_PI/k2;
		} 
		greens_kspace[i][1] = 0;	// Imaginary part.
	}
	
	// Calulate the density field from the particle positions.
	calc_density(p, density_real, NP, h); 
	
	// Print density field to a file. (DEBUGGING)
	fprint_density(density_real);
	
	// Forward Fourier transform of the density field. (2D ONLY HERE)
	fftw_execute(plan_forward);
	
	
	// Multiply density field with green's function in kspace.
	
	// Backwards Fourier transform to obtain the potential field.
	fftw_execute(plan_backward);
	
	
	// OTHER THINGS LATER
	
		
	// Free all dynamically allocated memory.
 	free(p);
 	
 	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
 	
 	fftw_free(density_real);
 	fftw_free(density_kspace);
 	fftw_free(greens_kspace);
 	fftw_free(pot_real);
 	
 	fftw_cleanup();
 	
 	return 0;
 }
