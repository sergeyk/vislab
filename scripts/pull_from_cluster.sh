#!/bin/zsh
echo "usage: pull_from_cluster <remote>"
REMOTE=$1

rsync -aP $REMOTE:work/vislab-git/data/shared/rqworkers ./data/shared/
rsync -aP $REMOTE:work/vislab-git/data/shared/predict ./data_shared/
rsync -aP $REMOTE:work/vislab-git/data/shared/redis ./data/shared/
rsync -aP $REMOTE:work/vislab-git/data/shared/db ./data/shared/

rsync -aP $REMOTE:work/vislab-git/data/mongodump ./data/
rsync -aP $REMOTE:work/vislab-git/data/results ./data/
