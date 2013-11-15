#!/bin/zsh

# rsync the repo to cluster

# code
rsync -ravzP --delete --exclude='*.pyc' --exclude='data' --exclude='vislab/tests/_temp' --prune-empty-dirs . flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab

# shared data
rsync -ravzP --delete data/shared/ flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab/data/shared
