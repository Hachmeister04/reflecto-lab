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

@lru_cache(maxsize=100)
def cached_full_profile_reconstruction(*args, **kwargs):
    kwargs['spectrogram_options'] = json.loads(kwargs['spectrogram_options'])
    kwargs['filters'] = json.loads(kwargs['filters'])
    return rpspy.full_profile_reconstruction(*args, **kwargs)

def get_dispersion_phase():  # Eventually to be replaced with a function from rpspy

  def phase(t, A, B, fc, f1, r):
      f = f1 + r * t
      return r * (A * t + B * f * (1 - (fc / f) ** 2) ** 0.5 / r)

  correction = np.exp(-1.0j * 2 * np.pi * phase(
    t=np.arange(1024) / 40e6, 
    # A= -5.45295e-08,  # LFS
    # B= 3.67217e-08,   # LFS
    A = -7.01476e-08,  # HFS
    B = 4.95304e-08,  # HFS
    fc=39.863*1e9,
    f1=48128186809.3756,
    r=1050746053958740.2
  ))
  return correction