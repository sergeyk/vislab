#!/bin/bash
mkdir -p data/shared/db/
srun -p vision --nodelist=flapjack1 --mem=188000 --time=0 --output=data/shared/db/job.log ./scripts/start_mongo.sh &
