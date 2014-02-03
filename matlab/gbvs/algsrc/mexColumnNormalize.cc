#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <matrix.h>
#include <string.h>

// Normalizes so that each column sums to one

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  //Declarations
  mxArray *Aar;
  double *A;
  double s;
  int i,j,numR,numC,myoff;

  // get first argument A
  Aar = (mxArray*)prhs[0];
  A = mxGetPr(Aar);
  numR = mxGetM(Aar);  // rows
  numC = mxGetN(Aar);  // cols

  for (j=0;j<numC;j++) {
    s = 0;    
    myoff = j*numR;
    for (i=0;i<numR;i++)
      s += A[ myoff + i ];
    for (i=0;i<numR;i++)
      A[ myoff + i ] /= s;
  }

  return;
}
