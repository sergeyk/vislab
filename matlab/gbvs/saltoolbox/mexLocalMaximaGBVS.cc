#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <matrix.h>
#include <string.h>

double getVal(double* img, int x, int y, int w, int h);
void getLocalMaxima(double* img, double thresh, int *lm_num, double *lm_sum, int w, int h);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  
  // input
  double* img, *thresh;
  
  // output
  double lm_avg, lm_sum; int lm_num;

  double* tmp;
  
  img = mxGetPr( prhs[0] );
  thresh = mxGetPr( prhs[1] );

  getLocalMaxima( img, thresh[0] , &lm_num, &lm_sum , mxGetN(prhs[0]) , mxGetM(prhs[0]) );
  
  if (lm_sum > 0) lm_avg = (double)lm_sum / (double)lm_num;
  else lm_avg = 0.0;
  
  plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL); //mxReal is our data-type
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL); //mxReal is our data-type
  plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL); //mxReal is our data-type
  
  tmp = mxGetPr(plhs[0]); tmp[0] = lm_avg;
  tmp = mxGetPr(plhs[1]); tmp[0] = lm_num;
  tmp = mxGetPr(plhs[2]); tmp[0] = lm_sum;
}

double getVal(double* img, int x, int y, int w, int h) 
{
  double* ptr = img + x * h + y;  
  return *ptr;
}

void getLocalMaxima(double* img, double thresh, int *lm_num, double *lm_sum, int w, int h)
{
  int i,j;
  double val;
  // then get the mean value of the local maxima:
  *lm_sum = 0.0; *lm_num = 0;
  
  for (j = 1; j < h - 1; j ++)
    for (i = 1; i < w - 1; i ++)
      {
        val = getVal(img,i,j,w,h);
        if (val >= thresh &&
            val >= getVal(img,i-1, j,w,h) &&
            val >= getVal(img,i+1, j,w,h) &&
            val >= getVal(img,i, j+1,w,h) &&
            val >= getVal(img,i, j-1,w,h))  // local max
          {
            *lm_sum += val;
            (*lm_num)++;
          }
      }
  return;
}


