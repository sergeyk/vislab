#!/bin/bash

codebook=$1
knn=$2
max_size=$3
step=$4
output=$5


if [ $# -lt 4 ]; then
  echo "not enough parameters"
  exit 1
fi

#export TMPDIR=/dev/shm  #change to your own temporary folder here

cd `dirname $0`;

local_in=`mktemp -d /tmp/tmp.XXXXXXXXXX` || exit 1
local_in=temp

local_out=$local_in.out

mkdir -p $local_out

tar -xf - -C $local_in || exit 1

mkdir -p $local_in.pgm || exit 1

echo "HERE"
echo `ls -C1 $local_in`

for i in `ls -C1 $local_in`; do
    #echo $i 1>&2
    convert -resize ${max_size}x${max_size}\> -depth 8 $local_in/$i $local_in.pgm/$i.pgm
    if [ $? -ne 0 ]; then
      rm -rf $local_in*;
      exit 1
    fi
done

cat >im2llc_script.m <<EOL
try
path(path,'third-party/vlfeat/toolbox');
path(path,'third-party/llc');

vl_setup;

imsizes = [ 1 0.5 0.25 ];
binSizes = [4 4 4];
steps = ceil($step * imsizes);

c = load('$codebook');

pyramid=[1,3];

files = dir('$local_in.pgm');

files = files(~[files.isdir]);

betas = [];
ids = cell(numel(files),1);

for i = 1:numel(files)

filename = files(i).name;
k = findstr(filename,'.');
id=filename(1:k(end-1)-1);

fprintf('extracting feature for %s\n', id);

I = im2single(imread(sprintf('$local_in.pgm/%s',filename)));

beta = im2llc(I,imsizes, binSizes, steps, c.codebook, pyramid, $knn, []);

if isempty(betas)
 betas = sparse(size(beta,1), numel(files));
 end

 betas(:,i) = beta;
 ids{i} = id;

 end

 save('$output','betas','ids', '-v7.3');

 system('touch $local_in.SUCCESS');

 catch ME

 ME
 ME.message
 ME.stack
 ME.stack.file
 ME.stack.name
 ME.stack.line

 exit

end
EOL

#matlab -nodesktop -nosplash -r im2llc_script

#if [ ! -e $local_in.SUCCESS ]; then
#  rm -rf $local_in*;
#  exit 1
#fi

#rm -rf $local_in*
