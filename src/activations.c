#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "wnet.h"

void activate_matrix(matrix m, ACTIVATION a){
  int i, j;
  for(i=0; i<m.rows; i++){
    double sum = 0;
    for(j=0; j<m.cols; j++){
      double x = m.data[i*m.cols +j];
      if(a==LOGISTIC){
        m.data[i*m.cols + j] = 1.0/ (1+exp(-x));
      } else if(a==RELU){
        m.data[i*m.cols + j] = x >0 ? x:0;
      } else if(a==LRELU) {
        m.data[i*m.cols + j] = x>0 ? x: 0.01*x;
      } else if(a==SOFTMAX) {
        m.data[i*m.cols + j] = exp(x);
      }
      else {
        fprintf(stderr, "activation error\n");
      }
      sum += m.data[i*m.cols +j];
    }
    if(a == SOFTMAX){
      for(int k=0; k<m.cols;k++){
        m.data[i*m.cols+k] /=sum;
      }
    }
  }
}

void gradient_matrix(matrix m, ACTIVATION a, matrix d){
  int i, j;
  for(i=0; i<m.rows; ++i){
    for(j=0;j<m.cols; ++j){
      double x = m.data[i*m.cols + j];
      if(a==LOGISTIC){
        d.data[i*m.cols +  j] *= x*(1-x);
      } else if(a==RELU){
        d.data[i*m.cols +j] *=x>0 ? 1:0;
      } else if(a==LRELU){
        d.data[i*m.cols +j] *= x>0? 1: 0.01;
      } else if(a==SOFTMAX || a==LINEAR){

      }
      else{
        fprintf(stdout, "action name error\n");
      }
    }
  }
}
