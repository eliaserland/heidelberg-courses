#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    // Determine machine epslion
    float em = 1;
    float em_last; 

    while (em + 1 != 1) {
        em_last = em; // Save old value.
        em = em * 0.5; // Divide and concour.
    }
    printf("\n Machine epsilon is %e \n\n", em_last);    

    return 0;
}









