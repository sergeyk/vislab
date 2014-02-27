#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <matrix.h>
#include <string.h>

// Avalues = mexArrangeLinear( A , dims )
// where A is the     NxM x K   matrix containing multi-resolution info
//       dims is      K x 2     matrix containing dimensions of each scale
//       Avalues is   P x 1 matrix containing multi-resolution info in flat array

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  //Declarations
  mxArray *Aar, *dimsar;
  double *A, *dims, *Avalues;

  int i,r,c,cur_index;
  int N, M, K, P, offset_,M_orig,N_orig,mapsize, coffset;

  // get first argument A
  Aar = (mxArray*)prhs[0];
  A = mxGetPr(Aar);

  // get sigma
  dimsar = (mxArray*)prhs[1];
  dims = mxGetPr(dimsar);
  K = mxGetM(dimsar); // number of rows

  P = 0;
  for (i=0;i<K;i++) 
    P += (int)( dims[ i ] * dims[ K + i ] );

  // create output
  plhs[0] = mxCreateDoubleMatrix(P, 1, mxREAL);
  Avalues = mxGetPr(plhs[0]);

  M_orig = (int)dims[0];
  N_orig = (int)dims[K];  
  mapsize = M_orig * N_orig;
  offset_ = 0;
  cur_index = 0;
  for (i=0;i<K;i++) {
    M = (int)dims[ i ];
    N = (int)dims[ K + i ];
    for (c=0;c<N;c++) {
      coffset = offset_ + c*M_orig;
      for (r=0;r<M;r++)
	Avalues[cur_index++] = A[ coffset + r ];
    }
    offset_ += mapsize;
  }

  return;
}
