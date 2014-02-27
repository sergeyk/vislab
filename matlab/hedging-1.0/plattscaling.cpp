#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>

#include "mex.h"

typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int l, const double *dec_values, const double *labels, 
	double& A, double& B)
{
	double prior1=0, prior0 = 0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;
	
	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=(double *)mxCalloc(l, sizeof(double));
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter; 
	
	// Initial Point and Initial Fun Value
	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
	double fval = 0.0;

	for (i=0;i<l;i++)
	{
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB = dec_values[i]*A+B;
		if (fApB>=0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++)
		{
			fApB = dec_values[i]*A+B;
			if (fApB >= 0)
			{
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step)
		{
			newA = A + stepsize * dA;
			newB = B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1+exp(-fApB));
				else
					newf += (t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
				A=newA;B=newB;fval=newf;
				break;
			}
			else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
      // Line search fails
			break;
		}
	}

	if (iter>=max_iter)
		mexPrintf("Reaching maximal iterations in two-class probability estimates\n");
	mxFree(t);
}


static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

void mexFunction( int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[] )
{
  if (nrhs != 2) {
    mexPrintf(
      "Usage: [a, b] = plattscaling(decision_value_vector, label_vector)\n"
      "a and b correspond to a sigmoid of 1 / (1 + exp(decision_value * a + b)).\n"
      "decision_value_vector should be a vector of doubles for a single class.\n"
      "label_vector should be a vector of doubles in {-1,1}.\n");
      fake_answer(plhs);
      return;
    }
	if (!mxIsDouble(prhs[0])) {
		mexPrintf("Error: decision value vector\n");
		fake_answer(plhs);
		return;
	}
	if (!mxIsDouble(prhs[1])) {
		mexPrintf("Error: label vector must be double\n");
		fake_answer(plhs);
		return;
	}
	if (!mxGetNumberOfElements(prhs[0]) || !mxGetNumberOfElements(prhs[1]) ||
      (mxGetNumberOfElements(prhs[0]) != mxGetNumberOfElements(prhs[1]))) {
		mexPrintf("Error: decision value and label vectors must have the same number of elements.\n");
		fake_answer(plhs);
		return;
	}
  // Get an array of decision values and labels from the input.
  const double *dec_values = mxGetPr(prhs[0]);
  const double *labels = mxGetPr(prhs[1]);
  // Assume the length is short enough that an int suffices.
  int length = (int)mxGetNumberOfElements(prhs[0]);
  double a;
  double b;
  // Learn a and b.
  sigmoid_train(length, dec_values, labels, a, b);
  // Write output.
  plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  *mxGetPr(plhs[0]) = a;
  *mxGetPr(plhs[1]) = b;
}

