#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

int main(int argc, char const *argv[])
{
        double x = 1e-8;
        
        /* by x = 0.000000015 = 1.5e-8, x+exp(-x) evaluates to 1, rounding away the first term.
           Then the numerator evalutes to zero, and the entire function becomes zero.  */        
        
        /* by x = 1e-6, cancellation becomes a serious factor as exp(-x) ~ 1.  
                We lose precision when exp(-x)-1 gets evaluated, the number is truncated, 
                and thus smaller than it should be. Smaller numerator => function 
                evaluates to larger than 0.5, which should in reality be impossible 
                for this function for positive x values. */

        /* by x = 1e-165, x*x overreaches the range of double, and we get division 
                by zero resulting in NaN. */ 
        
        double out0 = x;
        double out1 = exp(-x);
        double out2 = x + exp(-x);
        
        double out3 = x+exp(-x)-1;
        double out4 = x*x;
        double out5 = out3 / out4;

        printf("%-20.80g\n", out0);
        printf("%-20.80g\n\n", out1);
        printf("%-20.80g\n\n", out2);
        printf("%-20.80g\n\n", out3);
        printf("%-20.80g\n\n", out4);
        printf("%-20.80g\n\n", out5);
        return 0;
}
