#include "vector_broken.hh"
#include<iomanip>
#include<iostream>
#include<cstdlib>

void Vector::resize(int N_)
{
    entries.resize(N_);
    N = N_;
}

void Vector::resize(int N_, double value)
{
    entries.resize(N_);
    for (size_t i = 0; i < entries.size(); ++i)
    {
        entries[i] = value;
    }
    N = N_;
}

double& Vector::operator()(int i)
{
    if (i < 0 || i >= N)
    {
        std::cerr << "Illegal index " << i;
        std::cerr << " valid range is [0:" << N-1 << "]";
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
    return entries[i];
}

double Vector::operator()(int i) const
{
    if ( i < 0 || i >= N)
    {
        std::cerr << "Illegal index " << i;
        std::cerr << " valid range is [0:" << N-1 << "]";
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
    return entries[i];
}

double& Vector::operator[](int i)
{
    if (i < 0 || i >= N)
    {
        std::cerr << "Illegal index " << i;
        std::cerr << " valid range is [0:" << N-1 << "]";
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
    return entries[i];
}

double Vector::operator[](int i) const
{
    if (i < 0 || i >= N)
    {
        std::cerr << "Illegal index " << i;
        std::cerr << " valid range is [0:" << N-1 << "]";
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
    return entries[i];
}

Vector& Vector::operator*=(double x)
{
    for (int i = 0; i <= N; ++i)
        entries[i] *= x;
    return *this;
}

Vector& Vector::operator+=(const Vector& b)
{
  const int M = b.N;
    if (b.N != N)
    {
        std::cerr << "Dimensions of vector a (" << N
                  << ") and vector b (" << b.N
		  << ") do not match!";
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i <= N; ++i)
        entries[i] += b[i];
    return *this;
}

void Vector::print() const
{
    std::cout << "(" << N << ") vector:" << std::endl;
    for (int i = 0; i < N; ++i)
    {
        std::cout << std::setprecision(3);
	std::cout << std::setw(5) << entries[i] << " ";
    }
    std::cout << std::endl;
}

Vector operator+(const Vector& a, const Vector& b)
{
    Vector output(a);
    output += b;
    return output;
}

Vector operator*(const Vector& v, double x)
{
    Vector output(v);
    output *= x;
    return output;
}

Vector operator*(double x, const Vector& v)
{
    Vector output(v);
    output *= x;
    return output;
}

double operator*(const Vector& a, const Vector& b)
{
    const int N = a.size();
    if (b.size() != N)
    {
        std::cerr << "Dimensions of vector a (" << N
		  << ") and vector b (" << b.size()
		  << ") do not match!";
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
    double scalprod = 0.;
    for (int i = 0; i < N; ++i)
      scalprod += a(i) * b(i);
    return scalprod;
}

