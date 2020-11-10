#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>


double f(double x) {
        // If x closer to 0 than threshold, do
       
        return (x+exp(-x)-1)/(x*x);
}

int main(int argc, char const *argv[])
{
        // Allocations
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
                        // Print result
                        printf("f(x) = %g\n\n", fval);
                }
        }

        // Free dynamically allocated memory
        free(str);

        return 0;
}



