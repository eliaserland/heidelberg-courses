#include <stdio.h>
#include <stdlib.h>

/*
 * Fundamentals of Simulation Methods, Problem Set 1, Exercise 3.
 * Experimental determination of machine epsilon.
 * 
 * Author: Elias Olofsson   (ub253@stud.uni-heidelberg.de)
 * 
 * Version information: 
 *      2020-11-07: v.1.0. First public release.
 */

int main(int argc, char const *argv[])
{
    /* Determine machine epsilon for float, double and long double. Starting 
       with float. */
    float em_float = 1;
    float em_float_last; 

    /* Divide em ansatz in 2 until em + 1 = 1. Save the old value before each 
       division. */
    while (em_float + 1 != 1) {
        em_float_last = em_float; // Save old value.
        em_float = em_float * 0.5; // Divide.
    }

    printf("\nMachine epsilon for float is: \n%80.30e\n", em_float_last);
    printf("1 + e_m evaulates to: \n%80.50e\n\n\n", 1 + em_float_last);  

    /* Same as above, but for double. */
    double em_double = 1;
    double em_double_last;

    while (em_double + 1 != 1) {
        em_double_last = em_double; 
        em_double = em_double * 0.5;
    }

    printf("Machine epsilon for double is: \n%80.40le\n", em_double_last);
    printf("1 + e_m evaulates to: \n%80.60le\n\n\n", 1 + em_double_last);  

    /* Same as above, but for long double.*/
    long double em_long_double = 1;
    long double em_long_double_last;
    
    while (em_long_double + 1 != 1) {
        em_long_double_last = em_long_double;
        em_long_double = em_long_double * 0.5;
    }
    
    printf("Machine epsilon for long double is: \n%80.50Le\n", em_long_double_last);
    printf("1 + e_m evaulates to: \n%80.70Le\n\n\n", 1 + em_long_double_last);  

    return 0;
}









