#!/bin/bash
echo "usage: push_to_cluster <remote>"
echo "    rsyncs the repo to remote"
REMOTE=$1

# Code
rsync -aP --del --prune-empty-dirs --exclude='*.pyc' \
    --exclude "vislab/config.json" --exclude='data' --exclude='vislab/tests/_temp' \
    . $REMOTE:work/vislab-git

# Shared data
rsync -aP --del --prune-empty-dirs data/shared/ $REMOTE:work/vislab-git/data/shared
