
#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <matrix.h>
#include <string.h>

//  Vo = mexSumOverScales( v , lx , N )
//
//  name      dim       description
// -------------------------------------------------------------------------
//  v        P x 1      values of vector linearized
//  lx       P x (2+K)  K = lx(i,2)  # of locations corresponding to i
//                      lx(i,3:3+K)  individual locations corresponding to i
//  N        1 x 1      # of locations in original size map
//  Vo       N x 1      components of v summed and collapsed according to lx

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  // Declarations
  mxArray *var, *lxar, *Nar;
  double *v, *lx, *Nd, *Vo, vtmp;
  int i, j, K , N, P , P2, locum;

  // get v
  var = (mxArray*)prhs[0];
  v = mxGetPr(var);
  P = mxGetM(var);

  // get lx
  lxar = (mxArray*)prhs[1];
  lx = mxGetPr(lxar);

  // get N
  Nar = (mxArray*)prhs[2];
  Nd = mxGetPr(Nar);
  N = (int)Nd[0];

  // allocate Vo
  plhs[0] = mxCreateDoubleMatrix(N, 1, mxREAL);
  Vo = mxGetPr(plhs[0]);

  for (i=0;i<N;i++)
    Vo[i] = 0;

  P2 = 2 * P;
  for (i=0;i<P;i++) {
    K = (int)lx[ P + i ];
    vtmp = v[i] / (double)K;
    for (j=0;j<K;j++) {
      locum = (int)lx[ P2 + j*P + i ];
      Vo[ locum ] += vtmp;
    }
  }

  return;
}


