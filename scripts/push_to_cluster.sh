#!/bin/zsh

# rsync the repo to cluster

# code
rsync -ravzP --delete --exclude='*.pyc' --exclude='data' --prune-empty-dirs . flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab

# shared data
rsync -ravzP --delete data/shared flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab/data/
