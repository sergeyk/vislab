#!/bin/bash

mkdir -p data/db

numactl --interleave=all ls
if [ $? -ne 0 ]
then
    mongod --dbpath=data/db --logpath=data/db/mongod.log --port=27666
else
    numactl --interleave=all mongod --dbpath=data/db --logpath=data/db/mongod.log --port=27666
fi
