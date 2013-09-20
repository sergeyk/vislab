import os
import json

repo_dirname = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

with open(repo_dirname + '/backend/config.json', 'r') as f:
    config = json.load(f)
config['VOC_DIR'] = os.path.expanduser(config['VOC_DIR'])
config['data_dir'] = repo_dirname + '/data'
