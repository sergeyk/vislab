#!/bin/bash

# Make the database directory.
mkdir -p data/db

# Get the port from the config file.
PORT=`python -c "import json; print json.load(open('vislab/config.json'))['servers']['mongodb'][1];"`
CMD="mongod --dbpath=data/db --logpath=data/db/mongod.log --port=$PORT"

# On our Red Hat cluster, mongod wants to be launched in this mode.
numactl ls
if [ $? -eq 0 ]
    CMD="numactl --interleave=all $CMD"
fi

# Execute command.
CMD
