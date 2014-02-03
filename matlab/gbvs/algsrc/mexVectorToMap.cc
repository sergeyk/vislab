
#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <matrix.h>
#include <string.h>

//  outmap = mexVectorToMap( v , dim )
//
//  name      dim        description
// -------------------------------------------------------------------------
//  v        MN x 1      values of map in linear form
//  dim      1  x 2      =[M N] dimension of 2D map
//  outmap   M  x N      values of map in 2D map form

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  // Declarations
  mxArray *var, *dimar;
  double *v, *dim, *outmap;
  int i, M, N, MN;

  // get v
  var = (mxArray*)prhs[0];
  v = mxGetPr(var);

  // get dim
  dimar = (mxArray*)prhs[1];
  dim = mxGetPr(dimar);
  M = (int)dim[0];
  N = (int)dim[1];
  MN = M * N;

  // allocate outmap
  plhs[0] = mxCreateDoubleMatrix(M, N, mxREAL);
  outmap = mxGetPr(plhs[0]);

  for (i=0;i<MN;i++)
    outmap[i] = v[i];

  return;
}


