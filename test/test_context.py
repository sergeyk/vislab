import sys
import os
repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_dirname)

temp_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '_temp'))
if not os.path.exists(temp_dirname):
    try:
        os.makedirs(temp_dirname)
    except:
        pass
