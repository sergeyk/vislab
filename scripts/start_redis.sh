#!/bin/bash

# Get the port from the config file.
PORT=`python -c "import json; print json.load(open('vislab/config.json'))['servers']['redis'][1];"`

# Make the data directory and start the server in it.
mkdir -p data/redis
cd data/redis
redis-server --port $PORT
