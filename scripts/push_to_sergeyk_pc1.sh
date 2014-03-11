#!/bin/zsh

# rsync the repo to cluster
echo "Run on cluster"

# code
rsync -ravzP --delete --prune-empty-dirs \
    --exclude=".DS_Store" --exclude='*.pyc' --exclude='data' --exclude='vislab/tests/_temp' \
    . sergeyk-pc1:~/work/vislab

# shared data
rsync -ravzP --delete data/shared/ sergeyk-pc1:~/work/vislab/data/shared
