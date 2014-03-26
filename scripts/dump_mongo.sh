#!/bin/bash

mongodump --port 27666 --db predict --out data/mongodump

# DB_NAME="pinscraping"

# NAMES=`mongo $DB_NAME --eval "db.getCollectionNames()" | tail -n 1`
# IFS=$(echo -en "\n\b")
# for i in $(echo $NAMES | tr "," "\n")
# do
#   mongodump --db $DB_NAME --collection $i --out - | gzip > $DB_NAME/$i.bson.gz
# done
