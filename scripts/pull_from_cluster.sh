#!/bin/zsh

rsync -rtazP sergeyk@flapjack1.icsi.berkeley.edu:/u/sergeyk/work/vislab-git/data/shared/rqworkers ./data/shared/
rsync -rtazP sergeyk@flapjack1.icsi.berkeley.edu:/u/sergeyk/work/vislab-git/data/shared/predict ./data_shared/
rsync -rtazP sergeyk@flapjack1.icsi.berkeley.edu:/u/sergeyk/work/vislab-git/data/shared/db ./data/shared/
rsync -rtazP sergeyk@flapjack1.icsi.berkeley.edu:/u/sergeyk/work/vislab-git/data/shared/redis ./data/shared/
rsync -rtazP sergeyk@flapjack1.icsi.berkeley.edu:/u/sergeyk/work/vislab-git/data/results ./data/
rsync -rtazP sergeyk@flapjack1.icsi.berkeley.edu:/u/sergeyk/work/vislab-git/data/mongodump ./data/
