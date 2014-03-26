#!/bin/bash

echo "RUN THIS ONLY FROM FLAPJACK1 (or whatever the main store of feats is)!"
sleep 3

DIRNAME=/tscratch/tmp/sergeyk/vislab_feats

MACHINES="flapjack2,orange1,orange2,orange3,orange4,orange5,orange6,banana1,banana2,banana3,banana4"
echo $MACHINES | tr "," "\n" | parallel -j 4 rsync -trP --exclude '*.h5' $DIRNAME/ {}:$DIRNAME

rsync -rav $FEAT_DIR/ /u/vis/x1/sergeyk/vislab_feats &
