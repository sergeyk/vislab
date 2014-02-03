
function mask = makeFixationMask( X , Y , origimgsize , salmapsize )

%
% this maps (X,Y) fixation coordinates to fixation mask
%
% given fixation coordinates X and Y in original image coordinates,
% produces mask of same size salmapsize where each location contains
% an integer count of the fixations lying at that location
%

if ( length(X) ~= length(Y) )
  fprintf(2,'makeFixationMask Error: number of X and Y coordinates should be the same!\n');
  mask = [];
  return;
end

scale = salmapsize(1) / origimgsize(1);

X = round(X * scale);
Y = round(Y * scale);

X(X<1) = 1;
X(X>salmapsize(2)) = salmapsize(2);
Y(Y<1) = 1;
Y(Y>salmapsize(1)) = salmapsize(1);

mask = zeros( salmapsize );
for i = 1 : length(X)
  mask( Y(i) , X(i) ) = mask( Y(i) , X(i) ) + 1;
end