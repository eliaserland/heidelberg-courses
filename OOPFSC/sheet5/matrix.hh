#ifndef MATRIX_HH
#define MATRIX_HH

#include<vector>

class Matrix
{
  public:
    Matrix(int numRows_, int numCols_);
    Matrix(int dim);
    Matrix(int numRows_, int numCols_, double value);
    Matrix(const std::vector<std::vector<double> >& a);
    Matrix(const Matrix& b);
    void resize(int numRows_, int numCols_);
    void resize(int numRows_, int numCols_, double value);
    // access elements
    double& operator()(int i, int j);
    double  operator()(int i, int j) const;
    std::vector<double>& operator[](int i);
    const std::vector<double>& operator[](int i) const;
    // arithmetic functions
    Matrix& operator*=(double x);
    Matrix& operator+=(const Matrix& b);
    // output
    void print() const;
    int rows() const { return numRows; };
    int cols() const { return numCols; };
  private:
    std::vector<std::vector<double> > entries;
    int numRows = 0;
    int numCols = 0;
};

std::vector<double> operator*(const Matrix& a,
    const std::vector<double>& x);
Matrix operator*(const Matrix& a, double x);
Matrix operator*(double x, const Matrix& a);
Matrix operator+(const Matrix& a, const Matrix& b);

#endif // MATRIX_HH
