import os
import json
repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load config file.
with open(repo_dirname + '/config.json', 'r') as f:
    config = json.load(f)
for path_name, path in config['paths']:
    config['paths'][path_name] = os.path.expanduser(path)

# Set data directory in the config.
config['data_dir'] = repo_dirname + '/data'

import util
import datasets
import collection
