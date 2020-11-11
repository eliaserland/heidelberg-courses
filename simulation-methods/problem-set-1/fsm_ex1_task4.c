#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

/*
 * Fundamentals of Simulation Methods, Problem Set 1, Exercise 4.
 * Near-cancellation of floating point numbers.
 * 
 * Author: Elias Olofsson (ub253@stud.uni-heidelberg.de)
 * 
 * Version information: 
 *      2020-11-10: v.1.0. First public release.
 */

/*
 * f:   Analytical function f(x) = (x+exp(-1)+1)/x^2. For values x closer than a 
 *      certain threshold to x=0, a third order taylor expansion around x=0 is 
 *      used, instead of the analytical expression for f.
 */
double f(double x) 
{
        double out;
        double threshold = 0.001;
        if (fabs(x) < threshold) {
                out = 0.5 - x/6.0 + x*x/24.0 - x*x*x/120.0;
        } else {
                out = (x+exp(-x)-1)/(x*x);
        } 
        return out;
}

int main(int argc, char const *argv[])
{
        // Allocations.
        char *str = malloc(255);
        double x;
        double fval; 
        bool running = true;
        printf("Enter a real number x to calculate f(x), or type 'quit' to "
               "terminate program.\n");

        // While user has not chosen to exit program, repeat:
        while (running){
                // Get keyboard input.
                printf("  x  = ");
                fgets(str, 255, stdin);
                if (strncmp(str, "quit", 4) == 0) {
                        running = false;
                } else {
                        // Store input to variable x and calc f(x).
                        x = strtod(str, NULL);
                        fval = f(x);
                        // Print result.
                        printf("f(x) = %g\n\n", fval);
                }
        }

        // Free dynamically allocated memory.
        free(str);

        return 0;
}



