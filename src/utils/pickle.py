import pickle
import numpy as np
import os

def numpy_to_python(obj):
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(v) for v in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def read_pickle(filename):
    with open(filename, 'rb') as config_file:
        return pickle.load(config_file)

# def read_configs(filename, datasets_path=None, as_dict=False):
#     with open(filename, 'rb') as config_file:
#         configs = pickle.load(config_file)
#
#     if datasets_path is not None:
#         for c in configs:
#             c["background"] = c["background"].replace("./", datasets_path)
#
#     if as_dict:
#         return {c["image_id"]: c for c in configs}
#
#     return configs

def read_configs(filename, datasets_path=None, as_dict=False):
    with open(filename, 'rb') as config_file:
        configs = pickle.load(config_file)

    if datasets_path is not None:
        for c in configs:
            marker = "mc_gill_calibrated_colour_image_database"
            idx = c["background"].find(marker)
            if idx != -1:
                c["background"] = os.path.join(datasets_path, c["background"][idx:])

    if as_dict:
        return {c["image_id"]: c for c in configs}

    return configs


