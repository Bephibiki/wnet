#include "wnet.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "matrix.h"
#include "image.h"


matrix im2col(image im, int size, int stride){
  int outw = (im.w-1)/stride + 1;
  int outh = (im.h-1)/stride + 1;
  int cols = outh * outw;
  int rows = size * size * im.c;
  matrix out = make_matrix(rows, cols);

  int start = (size -1 )/2;

  int c_i, h_i, w_i;
  int x, y;
  float *im_channel_i;
  float *out_channel_i;
  for(c_i=0; c_i< im.c ; c_i++){
    im_channel_i = &im.data[c_i*im.h*im.w];
    out_channel_i = &out.data[c_i * size * size * cols];
    for(h_i = -start; h_i< im.h-start; h_i+=stride){
      for(x=0; x<size; x++){
        for(y=0; y<size; y++){
          for(w_i = -start + y; w_i < im.w-start; w_i+=stride ){
            float value = 0;
            if( (h_i+x ) < 0 || (h_i+x) >= im.h || w_i<0 || w_i>=im.w){
              value = 0;
            }
            else{
              value = im_channel_i[(h_i+x)*im.w + w_i];
            }
            out_channel_i[(x*size + y) * cols + ((h_i + start)/stride)*outw + (w_i-y+start)/stride] = value;
          }
        }
      }
    }
  }
  return out;
}

void col2im(matrix col, int size, int stride, image im){
  int outw = (im.w - 1) / stride + 1;
  int outh = (im.h - 1 )/stride + 1;
  // int rows = im.c * size * size;
  int cols = outw * outh;

  int start = (size -1 )/2;

  int c_i, h_i, w_i;
  int x, y;
  for(c_i=0; c_i< im.c ; c_i++){
    float *im_channel_i = &im.data[c_i*im.h*im.w];
    float *col_channel_i = &col.data[c_i * size * size * cols];
    for(h_i = -start; h_i< im.h-start; h_i+=stride){
      for(x=0; x<size; x++){
        for(y=0; y<size; y++){
          for(w_i = -start + y; w_i < im.w-start; w_i+=stride ){
            if( (h_i+x ) < 0 || (h_i+x) >= im.h || w_i<0 || w_i>=im.w){
              continue;
            }
            im_channel_i[(h_i+x)*im.w+w_i]+= col_channel_i[(x*size + y) * cols + ((h_i + start)/stride)*outw + (w_i+start-y)/stride] ;
          }
        }
      }
    }
  }
}

void forward_convolutional_bias(matrix m, matrix b){
  assert(m.cols %m.cols == 0);
  int spatial = m.cols/b.cols;
  int i, j;
  for(i=0; i<m.rows; i++){
    for(j=0; j<m.cols; j++){
      m.data[i*m.cols + j]+= b.data[j/spatial];
    }
  }
}

matrix forward_convolutional_layer(layer l, matrix in){
  int i, j;
  int outw =(l.width - 1)/l.stride + 1;
  int outh = (l.height - 1)/l.stride +1;
  matrix out = make_matrix(in.rows, outw*outh*l.filters);
  for(i=0; i<in.rows; ++i){
    image im = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
    matrix x = im2col(im, l.size, l.stride);
    // (filters, outw * outh)
    matrix wx = matmul(l.w, x);
    for(j=0; j<wx.rows *wx.cols ; ++j){
      out.data[i*out.cols +j] = wx.data[j];
    }
    free_matrix(x);
    free_matrix(wx);
  }
  if(l.batchnorm){
    matrix x_norm = batch_normalize_forward(l, out);
    out = x_norm;
  }
  forward_convolutional_bias(out, l.b);
  activate_matrix(out, l.activation);

  l.in[0] = in;
  free_matrix(l.out[0]);
  l.out[0] = out;
  free_matrix(l.delta[0]);
  l.delta[0] = make_matrix(out.rows, out.cols);
  return out;
}

void backward_convolutional_bias(matrix m, matrix db){
  assert(db.rows ==1);
  assert(m.cols % db.cols ==0);
  int spatial = m.cols / db.cols;
  int i, j;
  for(i=0; i<m.rows; i++){
    for(j=0; j<m.cols; ++j){
      db.data[j/spatial] += m.data[i*m.cols + j];
    }
  }
}


void backward_convolutional_layer(layer l, matrix prev_delta){
  matrix in = l.in[0];
  matrix out = l.out[0];
  matrix delta = l.delta[0];

  int outw = (l.width -1)/ l.stride +1;
  int outh = (l.height -1)/ l.stride +1;
  gradient_matrix(out, l.activation, delta);
  backward_convolutional_bias(delta, l.db);
  if(l.batchnorm){
    matrix dx = batch_normalize_backward(l, delta);
    free_matrix(delta);
    l.delta[0] = dx;
    delta = l.delta[0];
  }

 matrix wt = transpose_matrix(l.w);
 assert(in.cols == l.width * l.height *l.channels);
 int i;
  for(i=0; i<in.rows; ++i){
    image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
    image dexample = float_to_image(prev_delta.data + i*in.cols, l.width, l.height, l.channels);
    delta.rows = l.filters;
    delta.cols = outw * outh;
    delta.data = l.delta[0].data + i* delta.rows * delta.cols;

    matrix x = im2col(example, l.size, l.stride);
    matrix xt = transpose_matrix(x);
    matrix dw = matmul(delta, xt);
    axpy_matrix(1, dw, l.dw);

    if(prev_delta.data){
      matrix col = matmul(wt, delta);
      col2im(col, l.size, l.stride, dexample);
      free_matrix(col);
    }
    free_matrix(x);
    free_matrix(xt);
    free_matrix(dw);
  }
  free_matrix(wt);
}

void update_convolutional_layer(layer l, float rate, float momentum, float decay){
  axpy_matrix(-decay, l.w, l.dw);
  axpy_matrix(rate,l.dw, l.w);
  scal_matrix(momentum, l.dw);
  axpy_matrix(rate, l.db, l.b);
}


layer make_convolutional_layer(int w, int h, int c, int filters, int size,int stride, ACTIVATION activation, int batchnorm){
  layer l = {0};
  l.width = w;
  l.height = h;
  l.channels = c;
  l.filters = filters;
  l.stride = stride;
  l.size = size;

  l.w = random_matrix(filters, size*size*c, sqrtf(2.f/(size*size*c)));
  l.dw = make_matrix(filters, size*size*c);

  l.b = make_matrix(1, filters);
  l.db = make_matrix(1, filters);
  l.in = calloc(1, sizeof(matrix));
  l.out = calloc(1, sizeof(matrix));
  l.delta = calloc(1, sizeof(matrix));

  l.activation = activation;
  l.forward = forward_convolutional_layer;
  l.backward = backward_convolutional_layer;
  l.update = update_convolutional_layer;
  l.batchnorm = batchnorm;
  if(batchnorm){
    l.x = calloc(1, sizeof(matrix));
    l.rolling_mean = make_matrix(1, filters);
    l.rolling_variance = make_matrix(1, filters);
  }
  return l;
}
