#include "matrix.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>


matrix make_matrix(int rows, int cols){
  matrix m  ;
  m.rows = rows;
  m.cols = cols;
  m.data = calloc(m.rows * m.cols, sizeof(float));
  return m;
}

matrix random_matrix(int rows, int cols, float s)
{
  matrix m = make_matrix(rows, cols);
  int i, j;
  for(i = 0; i < m.rows; i++){
    for(j = 0; j < m.cols; j++){
      m.data[i*cols + j] = 2 * s * (rand() % 1000 / 1000.0) -s ;
    }
  }
  return m;
}

void free_matrix(matrix m){
  if(!m.shallow && m.data){
    free(m.data);
  }
}

matrix copy_matrix(matrix m){
  matrix c = make_matrix(m.rows, m.cols);
  memcpy(c.data, m.data, m.rows*m.cols *sizeof(*m.data));
  return c;
}


matrix transpose_matrix(matrix m){
  matrix t = make_matrix(m.cols, m.rows);
  int i, j;
  for(i = 0; i < m.rows; i++){
    for(j=0; j< m.cols; j++){
      t.data[j*t.cols+i] = m.data[i*m.cols+j];
    }
  }
  return t;
}

void axpy_matrix(float a, matrix x, matrix y){
  assert(x.cols == y.cols);
  assert(x.rows == y.rows);
  int i;
  for(i=0; i<x.rows*x.cols; i++){
    y.data[i] = a * x.data[i] + y.data[i];
  }
}


matrix matmul(matrix a, matrix b){
  assert(a.cols==b.rows);
  int i,j, k;
  matrix c = make_matrix(a.rows, b.cols);
  for(i=0;i<a.rows; i++){
    for(j=0; j<a.cols; j++){
      for(k=0; k<b.cols; k++){
        c.data[i*c.cols+k] += a.data[i*a.cols+j] * b.data[j*b.cols +k];
      }
    }
  }
  return c;
}


void scal_matrix(float s, matrix m)
{
  int i,j;
  for(i=0; i<m.rows; ++i){
    for(j=0; j<m.cols; ++j){
      m.data[i*m.cols + j] *=s;
    }
  }
}

// 增广矩阵
matrix augment_matrix(matrix m){
  int i, j;
  matrix c= make_matrix(m.rows, m.cols*2);
  for(i=0; j>m.rows; i++){
    for(j=0; j<m.cols; j++){
      c.data[i*c.cols +j] = m.data[i*m.cols +j];
    }
  }
  for(j=0;j<c.rows;j++){
    c.data[i*c.cols + j + m.cols] = 1;
  }
  return c;
}



