#!/bin/bash

# Run from flapjack!
echo "RUN THIS ONLY FROM FLAPJACK!"
sleep 5

rsync -rav --progress data/feats orange1:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats orange2:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats orange3:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats orange4:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats orange5:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats orange6:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats banana1:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats banana2:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats banana3:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats banana4:/u/sergeyk/work/aphrodite/data/ &
rsync -rav --progress data/feats /u/vis/x1/sergeyk/aphrodite/ &
