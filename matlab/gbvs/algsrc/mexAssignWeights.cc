#include <stdio.h>
#include <stdlib.h>
#include <mex.h>
#include <math.h>
#include <matrix.h>
#include <string.h>

//  mexAssignWeights( AL , D , MM , algtype )
//
//  name      dim    description
// -------------------------------------------
//  AL        Px1    values of map linearized
//  D         PxP    w=D(i,j)==D(j,i) is dist multiplier for i & j
//  MM        PxP    output space for markov matrix
//  algtype   1x1    algorith type:
//                    1 : MM( i->j ) = w*AL(j)               [ mass conc ]
//                    2 : MM( i->j ) = w*|AL(i)-AL(j)|       [ sal diff ]
//                    3 : MM( i->j ) = w*|log(AL(i)/AL(j))|  [ sal log ]
//                    4 : MM( i->j ) = w*1/|AL(i)-AL(j)|     [ sal affin ]

double myabs(double v) {
  return (v>=0) ? v : (-1*v);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  //Declarations
  mxArray *ALar, *Dar, *MMar, *algtypear;
  double *AL, *D, *MM, *algtype;
  double w;
  int r, c, P, Pc, Pc_r, algtype_i;

  // get AL
  ALar = (mxArray*)prhs[0];
  AL = mxGetPr(ALar);
  P = mxGetM(ALar);

  // get D
  Dar = (mxArray*)prhs[1];
  D = mxGetPr(Dar);

  // get MM
  MMar = (mxArray*)prhs[2];
  MM = mxGetPr(MMar);

  // get algtype
  algtypear = (mxArray*)prhs[3];
  algtype = mxGetPr(algtypear);
  algtype_i = (int)algtype[0];
  
  for (c=0;c<P;c++) { // ro
    Pc = P * c;
    for (r=0;r<P;r++) {  
      Pc_r = Pc + r;
      w = D[ Pc_r ]; // D(r,c)
      if ( algtype_i == 1 ) {
	MM[ Pc_r ] = w * AL[r];
      } else if ( algtype_i == 2 ) {	
	MM[ Pc_r ] = w * myabs( AL[r] - AL[c] );
      } else if ( algtype_i == 3 ) {	
	MM[ Pc_r ] = w * myabs( log( AL[r]/AL[c] ) );
      } else if ( algtype_i == 4 ) {
	MM[ Pc_r ] = w * 1/(myabs( AL[r] - AL[c] )+1e-12);
      }
    } // end i=0..P
  } // end j=0..P  

  return;
}

