#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <math.h>
#include <time.h>

#include "pcg_basic.h"
#define M_PI 3.14159265358979323846

double myrand(pcg32_random_t *rngptr){
    return ldexp(pcg32_random_r(rngptr) ,-32);
}

/*
// auxiliary function to create a Gaussian random deviate
double gaussian_rnd(void) {
	// --- students ---

	// Genrate two numbers from flat distribution on [0,1).
	double u1 = random
	double u2 =

	double d = ldexp(pcg32_random_r(&myrng), -32);

	// Use Box-Muller transform to standard normal pdf. [mu=0, std=1] 
	return sqrt(-2*log(u1))*sin(2*M_PI*u2)
	// --- end ---
}
*/

int main(int argc, char** argv) {


        double array[10];

        // Declare and seed.
        pcg32_random_t rngptr;
        pcg32_srandom_r(&rngptr, time(NULL), (intptr_t)&rngptr);
        
        /*
        for (int i; i < 50; i++) {
                double u1 = myrand(&rngptr);
                double u2 = myrand(&rngptr);
                double d = sqrt(-2*log(u1))*sin(2*M_PI*u2);
                printf("%f\n",d);
        }
        */
       for (int i = 0; i < 10; i++) {
                      printf("%d\n", i%1);
       }


        return 0;
}



