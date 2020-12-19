#include "vector_broken.hh"
#include<iostream>

int main()
{   // define vector
  Vector a(6,0.);
    for (int i = 0; i < a.size(); ++i)  
        a(i) = 2.;
    Vector b(4);
    for (size_t i = 0; i < b.size(); ++i)  
        b[i] = 1. + double(i);
    // print vector
    a.print();
    b.print();
    Vector c(a);
    a = 2 * c;
    a.print();
    a = c * 2.;
    a.print();
    a = c + a;
    a.print();
    const double scal = a * c;
    std::cout << "The scalarproduct of vector a and vector c is "
	      << scal << std::endl;
    std::cout << std::endl;
    const Vector d(b);
    std::cout << "Element 1 of vector d is " << d(1) << std::endl;
    std::cout << std::endl;
    a.resize(5,0.);
    a[0] = a[4] = 5.;
    a[1] = a[3] = -4.;
    a[2] = 4.;
    a.print();
}
