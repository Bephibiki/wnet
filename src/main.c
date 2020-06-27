#include <stdio.h>
#include <stdlib.h>
#include "list.h"
#include "wnet.h"

void try_mnist(){
   data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels") ;
   data test = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels") ;
   net n= {0};
   n.layers = calloc(2, sizeof(layer));
   n.n = 2;
   n.layers[0] = make_connected_layer(784, 64, LRELU, 0);
   n.layers[1] = make_connected_layer(64, 10, SOFTMAX, 0);

   int batch = 256;
   int iters = 5000;
   float rate = 0.02;
   float momentum = 0.95;
   float decay = 0.0;

   train_image_classifier(n, train, batch, iters, rate, momentum, decay);
   printf("Training accuracy:%f\n", accuracy_net(n, train));
   printf("Testing accuracy:%f\n", accuracy_net(n, test));
}

void try_mnist_cnn(){
   data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels") ;
   data test = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels") ;
   net n= {0};
   n.layers = calloc(3, sizeof(layer));
   n.n = 3;
   n.layers[0] = make_convolutional_layer(28, 28, 1, 1, 5, 2, LRELU, 0);
   n.layers[1] = make_convolutional_layer(14, 14, 1, 8, 5, 2, LRELU, 0);
   n.layers[2] = make_connected_layer(392, 10, SOFTMAX, 0);

   int batch = 128;
   int iters = 5000;
   float rate = 0.01;
   float momentum = 0.9;
   float decay = 0.0005;

   train_image_classifier(n, train, batch, iters, rate, momentum, decay);
   printf("Training accuracy:%f\n", accuracy_net(n, train));
   printf("Testing accuracy:%f\n", accuracy_net(n, test));

}

void try_mnist_cnn_maxpool()
{
    data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels");
    data test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels");

    net n = {0};
    n.layers = calloc(8, sizeof(layer));
    n.n = 8;
    //n.layers[0] = make_connected_layer(784, 32, LRELU);
    n.layers[0] = make_convolutional_layer(28, 28, 1, 8, 3, 1, LRELU, 1);
    n.layers[1] = make_maxpool_layer(28, 28, 8, 3, 2);
    n.layers[2] = make_convolutional_layer(14, 14, 8, 16, 3, 1, LRELU, 1);
    n.layers[3] = make_maxpool_layer(14, 14, 16, 3, 2);
    n.layers[4] = make_convolutional_layer(7, 7, 16, 32, 3, 1, LRELU, 1);
    n.layers[5] = make_maxpool_layer(7, 7, 32, 3, 2);
    n.layers[6] = make_convolutional_layer(4, 4, 32, 32,  3, 1, LRELU, 1);
    n.layers[7] = make_connected_layer(512, 10, SOFTMAX, 0);

    int batch = 128;
    int iters = 5000;
    float rate = .01;
    float momentum = .9;
    float decay = .0005;

    train_image_classifier(n, train, batch, iters, rate, momentum, decay);
    printf("Training accuracy: %f\n", accuracy_net(n, train));
    printf("Testing  accuracy: %f\n", accuracy_net(n, test));
}


int main(){
  try_mnist_cnn_maxpool();
  return 0;
}
