#!/bin/bash
mkdir -p data/shared/db/
srun -p vision --nodelist=flapjack --mem=72000 --time=0 --output=data/shared/db/job.log ./scripts/start_mongo.sh &
