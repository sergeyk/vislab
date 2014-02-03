#!/bin/zsh

# rsync the repo to cluster

# code
rsync -ravzP --delete --prune-empty-dirs \
    --exclude=".DS_Store" --exclude='*.pyc' --exclude='data' --exclude='vislab/tests/_temp' \
    . sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab

# shared data
rsync -ravzP --delete data/shared/ sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab/data/shared
