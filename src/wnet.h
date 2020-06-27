#ifndef WNET_H
#define WNET_H

#include "matrix.h"
#include "image.h"



// layer
typedef enum{CONNECTED} LAYER_TYPE;

typedef enum{LINEAR, LOGISTIC, RELU, LRELU, SOFTMAX} ACTIVATION;

typedef struct layer {
  matrix *in;
  matrix *out;
  matrix *delta;

  // weight
  matrix w;
  matrix dw;

  // bias;
  matrix b;
  matrix db;

  //
  int width, height, channels;
  int size, stride, filters;
  
  // activation
  ACTIVATION activation;
  LAYER_TYPE type;

  // batch normal
  int batchnorm;
  matrix *x;
  matrix rolling_mean;
  matrix rolling_variance;
  matrix x_norm;


  // functions
  matrix (*forward) (struct layer, struct matrix);
  void (*backward) (struct layer, struct matrix);
  void (*update) (struct layer, float rate, float momentum, float decay);

} layer;

layer make_connected_layer(int inputs, int outputs, ACTIVATION activation, int batchnorm);
layer make_convolutional_layer(int w, int h, int c, int filters, int size, int stride, ACTIVATION activation, int batchnorm);
layer make_maxpool_layer(int w, int h, int c, int size, int stride);


// activation

void activate_matrix(matrix m, ACTIVATION activation);
void gradient_matrix(matrix m, ACTIVATION a, matrix d);

matrix batch_normalize_forward(layer l, matrix x);
matrix batch_normalize_backward(layer l, matrix d);

//
// net
typedef struct {
  layer *layers;
  int n;
} net;


matrix forward_net(net m, matrix X);
void backward_net(net m);
void update_net(net m, float rate, float momentum, float decay);

// data
typedef struct {
  matrix X;
  matrix y;
} data;

data load_image_classification_data(char *images,  char *label_file);
data random_batch(data d, int n);
char *fgetl(FILE *fp);
void free_data(data d);



// cassifier

void train_image_classifier(net n, data d, int batch, int iters, float rate, float momentum, float decay);
float accuracy_net(net m, data d);

#endif

