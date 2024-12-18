import os
import rpspy
import json
import numpy as np
from functools import lru_cache


def round_to_nearest(value: float, value_list: list):
    return min(value_list, key=lambda x: abs(x - value))


def get_shot_from_path(path: str):
    try:
        return int(os.path.basename(path))
    except ValueError:
        raise Exception("Wrong file name")


def get_path_from_shot(shot: int):
    return f"/shares/experiments/aug-rawfiles/RTR/{shot//10}/{shot}"


#TODO: Use this for other functions
#Cached versions of functions
@lru_cache(maxsize=50)
def cached_get_linearization(*args, **kwargs):
    return rpspy.get_linearization(*args, **kwargs)


@lru_cache(maxsize=50)
def cached_get_auto_linearization_from_shares(shot, band):
    reference_shot, reference_sweep = rpspy.get_linearization_reference(shot)
    return rpspy.get_linearization(reference_shot, reference_sweep, band)


@lru_cache(maxsize=100)
def cached_full_profile_reconstruction(*args, **kwargs):
    kwargs['spectrogram_options'] = json.loads(kwargs['spectrogram_options'])
    kwargs['filters'] = json.loads(kwargs['filters'])
    return rpspy.full_profile_reconstruction(*args, **kwargs)

