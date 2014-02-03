#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <matrix.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  //---Inside mexFunction---

  //Declarations

  mxArray *xData, *sData;
  double *xValues, *outArray, *sValues;

  int i,j,k,z;
  int rowLen, colLen;

  int M;
  int Mo2;
  double sumPix;
  double sumPixsq;
  double gamma = 2.0;
  double StoDelta = 1.0;
  double weight;
  double delta = 1.0;
  double p;
  double var;
  double ni;
  int kv;
  int zv;

  // get first argument x
  xData = (mxArray*)prhs[0];
  xValues = mxGetPr(xData);
  rowLen = mxGetN(xData);
  colLen = mxGetM(xData);
  
  // get second argument sigma
  sData = (mxArray*)prhs[1];
  sValues = mxGetPr(sData);
  M = (int)sValues[0];
  Mo2 = (int)(M/2);

  //Allocate memory and assign output pointer
  plhs[0] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL); //mxReal is our data-type

  //Get a pointer to the data space in our newly allocated memory
  outArray = mxGetPr(plhs[0]);

  //Copy matrix while multiplying each point by 2
  for(i=0;i<rowLen;i++) {
    for(j=0;j<colLen;j++) {      
      sumPix = 0;
      sumPixsq = 0;
      ni = 0;
      for (k=i-Mo2;k<=i+Mo2;k++) {
	for (z=j-Mo2;z<=j+Mo2;z++) {	  
	  kv = k;  zv = z;	  	  
	  /* if ( kv < 0 ) kv = rowLen + kv;
	  else if ( kv >= rowLen) kv = kv - rowLen;
	  if ( zv < 0 ) zv = colLen + zv;
	  else if ( zv >= colLen ) zv = zv - colLen; */	  
	  if ( kv < 0 || kv >= rowLen || zv < 0 || zv >= colLen ) continue;	  	  
	  p = xValues[(kv*colLen)+zv];
	  sumPix += p;
	  sumPixsq += p*p;
	  ni++;
	}
      }
      var = 0;
      if ( ni != 0 ) {
	sumPix /= ni;
	var = sumPixsq - ni * sumPix * sumPix;
	var /= ni;	
      }
      outArray[(i*colLen)+j] = var;
    }
  }
  return;
}
