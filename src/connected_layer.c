#include "wnet.h"

#include <stdlib.h>
#include <math.h>
#include <assert.h>


void forward_bias(matrix m, matrix b){
  assert(b.rows==1);
  assert(m.cols == b.cols);
  int i,j;
  for(i=0; i<m.rows; ++i){
    for(j=0; j<m.cols; ++j){
      m.data[i*m.cols +j] += b.data[j];
    }
  }
}

void backward_bias(matrix delta, matrix db){
  int i, j;
  assert (db.cols== delta.cols);
  for(i=0; i<delta.rows;++i){
    for(j=0; j<delta.cols ; ++j){
      db.data[j] += delta.data[i*delta.cols +j];
    }
  }
}

matrix forward_connected_layer(layer l, matrix in){
  matrix out = matmul(in, l.w);
  if(l.batchnorm){
    matrix x_norm = batch_normalize_forward(l, out);
    free_matrix(out);
    out = x_norm;
  }
  forward_bias(out, l.b);
  activate_matrix(out, l.activation);

  l.in[0] = in;
  free_matrix(l.out[0]);
  l.out[0] = out;
  free_matrix(l.delta[0]);
  l.delta[0] = make_matrix(out.rows, out.cols);
  return out;
}

void backward_connected_layer(layer l, matrix prev_delta){
  matrix in = l.in[0];
  matrix out = l.out[0];
  matrix delta = l.delta[0];
  // 对激活函数求导
  gradient_matrix(out, l.activation, delta);
  //
  backward_bias(delta, l.db);
  if(l.batchnorm){
    matrix dx = batch_normalize_backward(l, delta);
    free_matrix(delta);
    l.delta[0] = dx;
    delta = l.delta[0];
  }
  axpy_matrix(1.0, matmul(transpose_matrix(in), delta), l.dw);

  if(prev_delta.data){
    axpy_matrix(1.0, matmul(delta,transpose_matrix(l.w)), prev_delta);
  }
}

void update_connected_layer(layer l, float rate, float momentum, float decay){
  // decay 的作用
  axpy_matrix(-decay, l.w, l.dw);
  axpy_matrix(rate, l.dw, l.w);
  scal_matrix(momentum, l.dw);
  axpy_matrix(rate, l.db, l.b);
}

layer make_connected_layer(int inputs, int outputs, ACTIVATION activation, int batchnorm){
  layer l = {0};
  l.w = random_matrix(inputs, outputs, sqrtf(2.f/inputs));
  l.dw = make_matrix(inputs, outputs);
  l.b = make_matrix(1, outputs);
  l.db = make_matrix(1, outputs);

  l.in = calloc(1, sizeof(matrix));
  l.out = calloc(1, sizeof(matrix));
  l.delta = calloc(1, sizeof(matrix));
  l.activation = activation;
  l.forward = forward_connected_layer;
  l.backward = backward_connected_layer;
  l.update = update_connected_layer;
  l.batchnorm = batchnorm;
  if(batchnorm){
    l.x = calloc(1, sizeof(matrix));
    l.rolling_mean = make_matrix(1, outputs);
    l.rolling_variance = make_matrix(1, outputs);
  }
  return l;
}


