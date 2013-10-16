#!/bin/zsh

rsync -ravz sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab/data/shared/rqworkers ./data/shared/
rsync -ravz sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab/data/shared/predict ./data/shared/
rsync -ravz sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab/data/shared/db ./data/shared/
rsync -ravz sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/vislab/data/shared/redis ./data/shared/
