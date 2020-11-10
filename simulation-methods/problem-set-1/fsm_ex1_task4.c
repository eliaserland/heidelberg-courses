#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>


double f(double x) {
        
        return (x + exp(-x) - 1) / (x*x);
}

int main(int argc, char const *argv[])
{
        // Allocattions
        char *str = malloc(255);
        double x;
        double fval; 
        bool running = true;
        printf("Enter a real number x to calculate f(x), or type 'quit' to "
               "terminate program.\n");

        // While user has not chosen to exit program, do
        while (running){
                // Get keyboard input
                fgets(str, 255, stdin);
                if (strncmp(str, "quit", 4) == 0){
                        running = false;
                } else {
                        // Store input to variable x
                        x = strtod(str, NULL);
                        // Calculate f(x)
                        fval = f(x);
                        // Print result
                        printf("x = %g, f(x) = %g\n", x, fval);
                }
        }


                // If x closer to 0 than threshold, do

        // Repeat

        free(str);

        return 0;
}



