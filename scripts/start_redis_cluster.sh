#!/bin/bash
mkdir -p data_shared/redis
srun -p vision --nodelist=flapjack --mem=8000 --time=0 --output=data_shared/redis/job.log --chdir=./data/redis redis-server &
