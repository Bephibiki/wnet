#include "wnet.h"
#include "float.h"
#include "math.h"

matrix mean(matrix x, int spatial){
  matrix m = make_matrix(1, x.cols/spatial);
  int i, j;
  for(i=0; i<x.rows; i++){
    for(j=0; j<x.cols; j++){
      m.data[j/spatial] += x.data[i*x.cols + j];
    }
  }
  for(i=0; i<m.cols; i++){
    m.data[i] = m.data[i] / x.rows / spatial;
  }
  return m;
}

matrix variance(matrix x, matrix m, int spatial){
  matrix v = make_matrix(1, x.cols / spatial);
  int i, j;
  for(i=0; i < x.rows; i++ ){
    for(j=0; j< x.cols; j++){
      v.data[j/spatial] += powf((x.data[i*x.cols + j] - m.data[j/spatial]), 2);
    }
  }
  for(i=0; i<v.cols; i++){
    v.data[i] = v.data[i] / x.rows / spatial;
  }
  return v;
}

matrix normalize(matrix x, matrix m, matrix v, int spatial){
  matrix x_norm =  make_matrix(x.rows, x.cols);
  int i,j, c;
  for(i=0; i<x.rows; i++){
    for(c = 0; c< x.cols / spatial; c++){
      float mean = m.data[c];
      float stdev = sqrtf(v.data[c] + FLT_EPSILON);
      for(j=0; j<spatial; j++){
        int idx = i*x.cols + c*spatial+ j;
        x_norm.data[idx] = (x.data[idx] - mean)/ stdev;
      }
    }
  }
  return x_norm;
}


matrix batch_normalize_forward(layer l, matrix x){
  float s = .1;
  int spatial = x.cols / l.rolling_mean.cols;
  if(x.rows == 1){
    return normalize(x, l.rolling_mean, l.rolling_variance, spatial);
  }

  matrix m =  mean(x, spatial);
  matrix v = variance(x, m, spatial);
  matrix x_norm = normalize(x, m, v, spatial);

  scal_matrix(1-s, l.rolling_mean);
  axpy_matrix(s, m, l.rolling_mean);

  scal_matrix(1-s, l.rolling_mean);
  axpy_matrix(s, v, l.rolling_mean);
  free_matrix(m);
  free_matrix(v);
  
  free_matrix(l.x[0]);
  l.x[0] = x;
  return x_norm;
}

matrix delta_mean(matrix d, matrix v, int spatial){
  matrix dm = make_matrix(1, v.cols);
  int i, j;
  for(i=0; i< d.rows;i++){
    for(j=0; j<d.rows;j++){
      dm.data[j/spatial] += d.data[i*d.cols + j];
    }
  }
  for(i=0;i< dm.cols; i++){
    dm.data[i] = dm.data[i] * (-1.0)/sqrtf(v.data[i]+FLT_EPSILON) ;
  }
  return dm;
}

matrix delta_variance(matrix d, matrix x, matrix mean, matrix variance, int spatial){
  matrix dv = make_matrix(1, variance.cols);
  int i, j;
  for(i=0; i< d.rows;i++){
    for(j=0;j<d.cols; j++){
      dv.data[j/spatial] += d.data[i*x.cols + j] * (x.data[i*x.cols + j] - mean.data[j/spatial]);
    }
  }
  for(i=0;i<dv.cols;i++){
    dv.data[i] = dv.data[i] * (-0.5) * powf((variance.data[i] + FLT_EPSILON), -1.5);
  }
  return dv;
}


matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix mean, matrix variance, matrix x, int spatial){
  int i,j;
  matrix dx = make_matrix(d.rows, d.cols);
  for(i=0; i<x.rows;i++){
    for(j=0; j<x.cols;j++){
      float m = mean.data[j/spatial];
      float v = variance.data[j/spatial];
      float delta_m = dm.data[j/spatial];
      float delta_v = dv.data[j/spatial];
      int idx = i * x.cols + j;
      dx.data[idx] = d.data[idx] / sqrtf(v + FLT_EPSILON) + delta_v * 2.0 * (x.data[idx] -m ) / x.rows / spatial + delta_m / x.rows / spatial;
    }
  }
  return dx;
}

matrix batch_normalize_backward(layer l, matrix d){
  int spatial = d.cols / l.rolling_mean.cols;
  matrix x = l.x[0];
  matrix m = mean(x, spatial);
  matrix v = variance(x, m, spatial);

  matrix dm = delta_mean(d, v, spatial);
  matrix dv = delta_variance(d, x, m, v, spatial);
  matrix dx = delta_batch_norm(d, dm, dv, m, v, x, spatial);

  free_matrix(m);
  free_matrix(v);
  free_matrix(dm);
  free_matrix(dv);

  return dx;
}
