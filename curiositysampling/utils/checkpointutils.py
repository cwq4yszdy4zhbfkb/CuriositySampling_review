import pickle
import gzip
import json


def save_pickle_object(obj_to_save, path=None, fname=None):
    full_fname = fname + ".gzip"
    with gzip.open(path + "/" + full_fname, "wb") as f:
        pickle.dump(obj_to_save, f)

    return full_fname


def save_json_object(obj_to_save, path=None, fname=None):
    full_fname = fname + ".json"
    with open(path + "/" + full_fname, "w") as f:
        json.dump(obj_to_save, f, indent=4)

    return full_fname


def append_json_object(obj_to_update, path=None, fname=None):
    with open(path + "/" + fname + ".json", "r") as f:
        data = json.load(f)

    data.update(obj_to_update)

    with open(path + "/" + fname + ".json", "w") as f:
        json.dump(data, f, indent=4)
