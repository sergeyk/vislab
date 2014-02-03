
% cleanmex

cd util
mex -maci64 myContrast.cc ;
cd ../

cd algsrc
mex -maci64 mexArrangeLinear.cc ;
mex -maci64 mexAssignWeights.cc ;
mex -maci64 mexColumnNormalize.cc ;
mex -maci64 mexSumOverScales.cc ;
mex -maci64 mexVectorToMap.cc ;
cd ../

cd saltoolbox/
mex -maci64 mySubsample.cc ;
mex -maci64 mexLocalMaximaGBVS.cc ;
cd ../
