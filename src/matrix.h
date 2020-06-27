#ifndef MATRIX_H
#define MATRIX_H



typedef struct matrix{
  int rows;
  int cols;
  float *data;
  int shallow;
} matrix;

matrix make_matrix(int rows, int cols);
matrix random_matrix(int rows, int cols , float s);

void free_matrix(matrix m);

matrix copy_matrix(matrix m);

matrix matmul(matrix a, matrix b);
// 
matrix mathmm(matrix a, matrix b);

// y=ax + y
void axpy_matrix(float a, matrix x, matrix y);

void scal_matrix(float s, matrix m);

void print_matrix(matrix m);

matrix transpose_matrix(matrix m);

matrix augment_matrix(matrix m);



#endif
