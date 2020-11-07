#include <stdio.h>

int main(int argc, char const *argv[])
{

    float x = 1e20;
    float y;
    y =  x*x/x;
    printf("y = x*x/x = %e\n", y);
    printf("\nx = %e \ny/x = %e\n\n", x, y/x);

    
    /*
    float z = 1.0;
    for (long i = 0; i < 1e10; i++)
    {
        z += 1.0;
    }
    
    printf("z = %e \n", z);
    */

    /*
    double a = 1.0e17;
    double b = -1.0e17;
    double c = 1.0;

    double x = (a + b) + c;
    double y = a + (b + c);

    printf("\nx = %e  \ny = %e\n\n", x, y); 
    */

    return 0;
}