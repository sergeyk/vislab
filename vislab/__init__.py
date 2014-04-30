import os
import json
repo_dirname = os.path.abspath(os.path.dirname(__file__))

# Load config file.
with open(repo_dirname + '/config.json', 'r') as f:
    config = json.load(f)
for path_name, path in config['paths'].iteritems():
    config['paths'][path_name] = os.path.expanduser(path)

import util
# import feature
