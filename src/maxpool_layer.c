#include "wnet.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

matrix forward_maxpool_layer(layer l, matrix in){
  int outw = (l.width - 1) / l.stride + 1; 
  int outh = (l.height - 1) / l.stride + 1;
  int im_i, c_i, h_i, w_i;
  int x, y;
  int start = (l.stride -1 ) / 2;
  float *in_im_ptr;
  float *in_channel_ptr;
  float *out_im_ptr;
  float *out_channel_ptr;

  matrix out = make_matrix(in.rows, outw * outh * l.channels);
  for(im_i= 0; im_i < in.rows; im_i++){
    in_im_ptr = &in.data[im_i * in.cols];
    out_im_ptr = &out.data[im_i * out.cols];
    for(c_i = 0; c_i < l.channels; c_i++){
      in_channel_ptr = &in_im_ptr[c_i * l.width * l.height];
      out_channel_ptr = &out_im_ptr[c_i * outw * outh];
      for(h_i = -start; h_i < l.height - start; h_i += l.stride){
        for(w_i = -start ; w_i < l.width - start; w_i += l.stride){
          float max_pixel = -FLT_MAX;
          for(x=0; x < l.size; x++){
            for(y=0; y < l.size; y++){
              int im_col = h_i + y;
              int im_row = w_i + x;
              if(im_col < 0 || im_col >= l.height || im_row < 0 || im_row > l.width){
                continue;
              }
              float pixel = in_channel_ptr[im_row * l.width + im_col] ;
              max_pixel = max_pixel > pixel ? max_pixel: pixel;
            }
          }
          int out_col = (w_i + start) / l.stride;
          int out_row = (h_i + start) / l.stride;
          out_channel_ptr[out_row * outw + out_col] = max_pixel;
        }
      }
    }
  }
  l.in[0] = in;
  free_matrix(l.out[0]);
  l.out[0] = out;
  free_matrix(l.delta[0]);
  l.delta[0] = make_matrix(out.rows, out.cols);
  return out;
}


void backward_maxpool_layer(layer l, matrix prev_delta){
  matrix in = l.in[0];
  matrix delta = l.delta[0];
  int outw = (l.width - 1) / l.stride + 1;
  int outh = (l.height - 1) / l.stride + 1;
  int start = (l.stride - 1) / 2;

  int im_i, c_i, h_i, w_i;
  int x, y;
  float *in_im, *in_channel;
  float *delta_im, *delta_channel;
  float *prev_delta_im, *prev_delta_channel;
  for(im_i = 0; im_i < in.rows; im_i++){
    in_im = &in.data[im_i * in.cols];
    delta_im = &delta.data[im_i * delta.cols];
    prev_delta_im = &prev_delta.data[im_i * prev_delta.cols];
    for(c_i = 0; c_i < l.channels; c_i++ ){
      in_channel = &in_im[c_i * l.width * l.height];
      delta_channel = &delta_im[c_i * outw * outh];
      prev_delta_channel = &prev_delta_im[c_i * l.width* l.height];
      for(h_i = -start; h_i < l.height - start; h_i += l.stride ){
        for(w_i = -start; w_i < l.width - start; w_i += l.stride){
          int max_im_col = -1;
          int max_im_row = -1;
          float max_pixel = -FLT_MAX;
          for(x = 0; x< l.size; x++){
            for(y = 0; y < l.size; y++){
              int im_col = w_i + x;
              int im_row = h_i + y;
              if(im_col < 0  || im_col >= l.width || im_row < 0 || im_row >= l.height){
                continue;
              }
              float pixel = in_channel[im_row * l.width + im_col] ;
              if(pixel > max_pixel){
                max_pixel = pixel;
                max_im_col = im_col;
                max_im_row = im_row;
              }
            }
          }
          int out_col = (w_i + start) / l.stride;
          int out_row = (h_i + start) / l.stride;
          prev_delta_channel[max_im_row * l.width + max_im_col] += delta_channel[out_row * outw + out_col];
        }
      }
    }
  }
}


void update_maxpool_layer(layer l, float rate, float momentum, float decay){
}


layer make_maxpool_layer(int width, int height, int channels, int size, int stride){
  layer l = {0};

  l.width = width;
  l.height = height;
  l.channels = channels;
  l.size = size;
  l.stride = stride;
  l.in = calloc(1, sizeof(matrix));
  l.out = calloc(1, sizeof(matrix));
  l.delta = calloc(1, sizeof(matrix));
  l.forward = forward_maxpool_layer;
  l.backward = backward_maxpool_layer;
  l.update = update_maxpool_layer;
  return l;
}
