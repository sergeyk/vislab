
% cleanmex

cd util
mex('myContrast.cc');
cd ../

cd algsrc
mex('mexArrangeLinear.cc');
mex('mexAssignWeights.cc');
mex('mexColumnNormalize.cc');
mex('mexSumOverScales.cc');
mex('mexVectorToMap.cc');
cd ../

cd saltoolbox/
mex('mySubsample.cc');
mex('mexLocalMaximaGBVS.cc');
cd ../
