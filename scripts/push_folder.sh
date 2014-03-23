#!/bin/bash

echo "RUN THIS ONLY FROM FLAPJACK!"
echo "usage: sync_folder /tscratch/tmp/dirname/"
echo "will sync contents to same directory on all oranges and bananas."
echo "don't forget the ending /!"
sleep 3

DIRNAME=$1
MACHINES="flapjack2,orange1,orange2,orange3,orange4,orange5,orange6,banana1,banana2,banana3,banana4"
echo $MACHINES | tr "," "\n" | parallel -j 8 ssh {} mkdir -p $DIRNAME
echo $MACHINES | tr "," "\n" | parallel -j 8 rsync -trv --exclude '*.h5' $DIRNAME {}:$DIRNAME
