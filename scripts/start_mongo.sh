#!/bin/bash

# Make the database directory.
mkdir -p data/db

# Get the port from the config file.
PORT=`python -c "import json; print json.load(open('vislab/config.json'))['servers']['mongo'][1];"`
CMD="mongod --dbpath=data/db --logpath=data/db/mongod.log --port $PORT"

# On our Red Hat cluster, mongod wants to be launched in this mode.
numactl ls > /dev/null 2> /dev/null
if [ $? -eq 0 ]; then
    CMD="numactl --interleave=all $CMD"
fi

# Execute command.
echo "Running: $CMD"
$CMD
