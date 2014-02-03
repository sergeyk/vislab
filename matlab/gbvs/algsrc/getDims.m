function [ dims ] = getDims( orig_size , deltas )
[tmp,dims] = formMapPyramid( ones(orig_size) , deltas );



