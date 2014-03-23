#!/bin/bash
mkdir -p data/shared/redis
srun -p vision --nodelist=flapjack1 --mem=48000 --time=0 --output=data/shared/redis/job.log --chdir=./data/redis redis-server &
