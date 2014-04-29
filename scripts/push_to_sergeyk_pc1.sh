#!/bin/zsh

# rsync the repo to cluster
# code
rsync -ravzP --delete --prune-empty-dirs \
    --exclude=".DS_Store" --exclude='*.pyc' --exclude='data' --exclude='vislab/tests/_temp' \
    . sergeyk-pc1.vicsi:~/work/vislab-git

# shared data
rsync -ravzP --delete data/shared/ sergeyk-pc1.vicsi:~/work/vislab-git/data/shared
