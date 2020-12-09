/*************************************/
/* NVT ensemble with neighbours list */
/*************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <math.h>
#include <time.h>

// PCG Random Number Genrator (https://www.pcg-random.org/).
#include "pcg_basic.h"

#define MAXPART 10000
#define M_PI 3.14159265358979323846

// Data type for storing the information for the particles.
typedef struct {
	double pos[3];
	double vel[3];
	double acc[3];
	double acc_prev[3];
	double pot;
	int n_neighbors; 	// Length of neighbor list
	int neighbors[MAXPART]; // Neighbors (with index larger than current particle)
} particle;


// Auxiliary function to create a Gaussian random deviate.
double gaussian_rnd(pcg32_random_t *rngptr) {
	// --- students ---
	// Genrate two numbers from flat distribution on [0,1).
	double u1 = ldexp(pcg32_random_r(rngptr) ,-32);
	double u2 = ldexp(pcg32_random_r(rngptr) ,-32);

	// Use Box-Müller transform to standard normal pdf (mu=0, std=1). 
	return sqrt(-2*log(u1))*sin(2*M_PI*u2);
	// --- end ---
}

// Create (or re-construct) neighbor lists (with index larger than current particle).
void create_neighborlist(particle *p, int N_tot, double boxsize, double neighbor_cut) {
	
	int n_count;
	double dr[3];
	double neighbor_cut2 = neighbor_cut * neighbor_cut;

	/* Reset all neighbour lists (only the length needs to be reset, we can 
	   leave the actual arrays of neigbor indicies). */
	for (int i = 0; i < N_tot; i++) {
		p[i].n_neighbors = 0;
	}

	for (int i = 0; i < N_tot-1; i++) {
		for (int j = i+1; j < N_tot; j++) {
			
			// Calculate squared distance from i to j.
      			double r2 = 0;
			for (int k = 0; k < 3; k++) {
				dr[k] = p[i].pos[k] - p[j].pos[k];

				// Ensure we find the shortest distance bewteen particles
				if (dr[k] > 0.5 * boxsize) {
					dr[k] -= boxsize;
				}
				if (dr[k] < -0.5 * boxsize) {
					dr[k] += boxsize;
				}
				r2 += dr[k] * dr[k];
			}
			// If within neighbor radius, add j to i's neighbor list.
			if (r2 < neighbor_cut2) {
				n_count = p[i].n_neighbors;
				p[i].neighbors[n_count] = j;
				p[i].n_neighbors += 1;
			}
		}
	}
}


// This function initializes our particle set.
void initialize(particle *p, double L, int N1d, double sigma_v, double neighbor_cut) {
	int n = 0;
	double dl = L / N1d;

	// Declare and seed the RNG.
	pcg32_random_t rngptr;
	pcg32_srandom_r(&rngptr, time(NULL), (intptr_t)&rngptr);

	for(int i = 0; i < N1d; i++) {
	    	for(int j = 0; j < N1d; j++) {
			for(int k = 0; k < N1d; k++) {
			// --- students ---
			p[n].pos[0] = (0.5+k)*dl;
			p[n].pos[1] = (0.5+j)*dl;
			p[n].pos[2] = (0.5+i)*dl;

			p[n].vel[0] = sigma_v*gaussian_rnd(&rngptr);
			p[n].vel[1] = sigma_v*gaussian_rnd(&rngptr);
			p[n].vel[2] = sigma_v*gaussian_rnd(&rngptr);

			for (int m = 0; m < 3; m++) {
				p[n].acc[m] = 0;
				p[n].acc_prev[m] = 0;
			}
			// --- end ---
			n++;
			}
		}
  	}
	// Create neighbor lists.
	int N_tot = N1d * N1d * N1d;
	create_neighborlist(p, N_tot, L, neighbor_cut);
}

/* This function updates the velocities by applying the accelerations for the 
   given time interval. */
void kick(particle * p, int ntot, double dt) {
	// --- students ---
	// For all particles and each dimension.
	for (int i = 0; i < ntot; i++){
		for (int k = 0; k < 3; k++) {
			// Kick particle (Velocity-Verlet).
			p[i].vel[k] += dt/2*(p[i].acc[k] + p[i].acc_prev[k]);
		}
	}
	// --- end ---
}

// This function drifts the particles with their velocities for the given time interval.
// Afterwards, the particles are mapped periodically back to the box if needed.
void drift(particle * p, int ntot, double boxsize, double dt) {
	// --- students ---
	// For all particles and each dimension.
	for (int i = 0; i < ntot; i++){
		for (int k = 0; k < 3; k++) {
			// Drift position (Velocity-Verlet).
			p[i].pos[k] += dt*p[i].vel[k] + dt*dt/2*p[i].acc[k];

			// If outside box, map back.
			if (p[i].pos[k] >= boxsize) {
				p[i].pos[k] -= boxsize; 
			}
			if (p[i].pos[k] < 0) {
				p[i].pos[k] += boxsize;
			}
		}
	}
	// --- end ---
}

// This function calculates the potentials and forces for all particles. For simplicity,
// we do this by going through all particle pairs i-j, and then adding the contributions both to i and j.
void calc_forces(particle * p, int ntot, double boxsize, double rcut)
{
	int n;
	double rcut2 = rcut * rcut;
	double r2, r6, r12, dr[3], acc[3], pot;

	// Store the accelerations of the previous step, and then set all the 
	// current accelerations and potentials to zero.
	for (int i = 0; i < ntot; i++) {
		p[i].pot = 0;
    		for (int k = 0; k < 3; k++) {
			p[i].acc_prev[k] = p[i].acc[k]; 
			p[i].acc[k] = 0;
		}
  	}

	// Sum over all distinct pairs.
	for (int i = 0; i < ntot; i++) {
		for (n = 0; n < p[i].n_neighbors; n++) {
			int j = p[i].neighbors[n];

			// Calculate squared distance.
			r2 = 0;
			for (int k = 0; k < 3; k++) {
				dr[k] = p[i].pos[k] - p[j].pos[k];

				// Ensure we find the shortest distance bewteen particles
				if (dr[k] > 0.5 * boxsize) {
					dr[k] -= boxsize;
				}
				if (dr[k] < -0.5 * boxsize) {
					dr[k] += boxsize;
				}
				r2 += dr[k] * dr[k];
			}

			// --- students ---
			if (r2 < rcut2) {
				r6 = r2 * r2 * r2;
				r12 = r6 * r6;

				// Calculate the Lennard-Jones potential for the pair.
				pot = 4*(1/r12-1/r6);
				p[i].pot += pot;
				p[j].pot += pot;

				// Calculate the Lennard-Jones force between the particles.
				for (int k = 0; k < 3; k++) {
					acc[k] = 24/r2*(2/r12-1/r6)*dr[k];
					p[i].acc[k] += acc[k];
					p[j].acc[k] -= acc[k];
				}
			}
			// --- end ---
		}
  	}
}

// This function calculates the total kinetic and total potential energy, averaged per particle.
void calc_energies(particle *p, int ntot, double *ekin, double *epot) {
	double sum_pot = 0, sum_kin = 0;

  	for (int i = 0; i < ntot; i++) {
    		sum_pot += p[i].pot;
		// --- students ---
		for(int k = 0; k < 3; k++) {
			sum_kin += p[i].vel[k] * p[i].vel[k];
		// --- end ---
		}
  	}
  	*ekin = 0.5 * sum_kin / ntot;
  	*epot = 0.5 * sum_pot / ntot;
}



/*
 * Main driver routine.
 */
int main(int argc, char **argv) {
	
	// Input parameters.
	double target_temperature = 80; // Target temperature
	double sig_v = sqrt(target_temperature / 120.0);
	int N1d = 5; 		 // Particles per dimension
	int N = N1d * N1d * N1d; // Total number of particles.

	// Box parameters.
	double L = 5.0 * N1d;
	double rcut = L; // or 5.0
	double boxsize = L;

	// Time control.
	int output_frequency = 10;
	int nsteps = 10000; 	// Number of iterations
	double dt = 0.01; 	// Timestep size

	// Neighbor lists.
	double neighbor_cut = 2*rcut; // Cutoff radius for neighbor lists (must be larger than rcut).
	int list_freq = 1;	 // Re-build neighbor lists every n:th iteration.

	/* If neighbor cutoff is large, all neighborlists will remain unchanged. 
	   Avoid unnecessary computations by building lists only once. */ 
	if (neighbor_cut > sqrt(3)/2*L) {
		list_freq = nsteps;
	}

	double ekin; 
	double epot;
	double t = 0;

	// Allocate storage for our particles.
	particle *p = malloc(N * sizeof(particle));

	// Initialize the particles.
	initialize(p, L, N1d, sig_v, neighbor_cut);

	// Calculate the forces at t=0.
	calc_forces(p, N, boxsize, rcut);
	
	// Create an output file.
	char fname[100];
	sprintf(fname, "output_T%d.txt", (int)target_temperature);
	FILE *fd = fopen(fname, "w");

	// Measure energies at beginning, and output this to the file and screen.
	calc_energies(p, N, &ekin, &epot);
	fprintf(fd, "%6f %13.8e %13.8e %13.8e\n", t, ekin, epot, ekin + epot);

	printf("nsteps: %d, dt: %.4f, T: %.4f \n", nsteps, dt, target_temperature);

	// --- students ---
	

	// Carry out the time integration using leapfrog integration.
	//clock_t tic = clock();
	for (int step = 0; step < nsteps; step++) {
		
		// Full step in positions (Velocity-Verlet). 
		drift(p, N, boxsize, dt); 

		// Update accelerations at the new positions.
		calc_forces(p, N, boxsize, rcut);

		// Full step in velocity (Velocity-Verlet).
		kick(p, N, dt);	

		// Update time.
		t += dt;

		// Re-construct neighbor lists.
		if (step % list_freq == 0) {
			create_neighborlist(p, N, boxsize, neighbor_cut);
		}

		// Calculate kinetic, potential and total energies.
		if (step % output_frequency == 0) {
			calc_energies(p, N, &ekin, &epot);
			fprintf(fd, "%6f %13.8e %13.8e %13.8e\n", t, ekin, epot, ekin + epot);
		}
	}

	// Free dynamically allocated memory.
	free(p);

	// Close file.
	fclose(fd);

	// --- end ---
	printf("boxsize = %.3f\n", boxsize);

	return 0;
}