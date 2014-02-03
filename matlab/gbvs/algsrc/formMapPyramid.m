
%
% for each delta in deltas , adds map 
% delta binary orders smaller to A
% stacks dimensions of maps in A
%

function [ Apyr , dims ] = formMapPyramid( A , deltas )

my_eps = 1e-12;
       
num_deltas = length(deltas);
max_delta = max(deltas);
num_pyr = 1 + num_deltas;

dim = [ size(A) num_pyr ];
Apyr = zeros( dim );
dims = zeros(num_pyr,2);

maps = {};
maps{1}.map = A;
maps{1}.siz = size(A);
last = A;

for i=1:max_delta;
    last = mySubsample( last );
    maps{i+1}.map = last;
    maps{i+1}.siz = size(last); 
end

for i=1:max_delta+1
   map = maps{i}.map;
   map = mat2gray(map);
   if ( max(map(:)) == 0 ) map = map + my_eps; end
   maps{i}.map = map;
end

Apyr(:,:,1) = maps{1}.map;
dims(1,:) = maps{1}.siz;

if ( size(deltas,1) > size(deltas,2) ) deltas = deltas'; end
i = 1;
for delta=deltas,    
    i = i + 1;
    d = maps{1+delta}.siz;
    m = maps{1+delta}.map;    
    Apyr(1:d(1),1:d(2),i) = m;
    dims(i,:) = d;
end


