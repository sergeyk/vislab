#!/bin/bash

input=$1
output=$2

knn=5
codebook=codebook.mat
max_size=500
step=4

./im2llc.sh $codebook $knn $max_size $step $output < $input

