#!/bin/zsh

rsync -ravz sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/aphrodite/data_shared/rqworkers ./data_shared/
rsync -ravz sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/aphrodite/data_shared/predict ./data_shared/
rsync -ravz sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/aphrodite/data_shared/db ./data_shared/
rsync -ravz sergeyk@flapjack.icsi.berkeley.edu:/u/sergeyk/work/aphrodite/data_shared/redis ./data_shared/
