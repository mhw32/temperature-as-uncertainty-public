#
# 1. Load all json files under config
# 2. change exp_base to null
# 3. if model, encode, exp_dir, change exp_dir and checkpoint to null
#

import json
import os

import pudb; pudb.set_trace()


def get_file_paths(dir_path):
    paths = []
    for file_path in os.listdir(dir_path):
        path = os.path.join(dir_path, file_path)
        if 'json' in file_path:
            paths.append(path)
        elif os.path.isdir(path):
            paths += get_file_paths(path)
    return paths

paths = get_file_paths('config')

for path in paths:
    with open(path, 'r') as f:
        config = json.load(f)
        config['exp_base'] = None
        if 'model' in config:
            if 'encoder' in config['model']:
                if 'exp_dir' in config['model']['encoder']:
                    config['model']['encoder']['exp_dir'] = None
                    config['model']['encoder']['checkpoint_name'] = None
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)





