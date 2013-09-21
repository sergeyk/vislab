#!/bin/zsh

# rsync the repo to cluster

# code
rsync -ravz --delete --exclude=".DS_Store" --exclude='*.pyc' --exclude='data' --prune-empty-dirs . flapjack.icsi.berkeley.edu:/u/sergeyk/work/aphrodite

# results data
rsync -ravz --delete data/results flapjack.icsi.berkeley.edu:/u/sergeyk/work/aphrodite/data/
