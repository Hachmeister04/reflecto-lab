import os
import rpspy
import json
import numpy as np
from functools import lru_cache
from scipy.interpolate import RegularGridInterpolator

from constants import DEFAULT_LINEARIZATION_SWEEP

try:
    from ipfnpytools.trz_to_rhop import fast_trz_to_rhop
    import aug_sfutils as sf
    AUG_MODE = True
except ImportError:
    print("Warning: ipfnpytools and aug_sfutils not found. Some functions may not work.")
    AUG_MODE = False


@lru_cache(maxsize=5)
def cached_get_equilibrium_interpolator(shot):
    """Convert time, rho poloidal, and z coordinates to R.

    Parameters
    ----------
    shot: int
        Shot number.

    Returns
    -------
    rho_trz: RegularGridInterpolator
        Rho poloidal interpolator.
    """
    eq = sf.EQU(shot, diag='IDE')
    if not eq.sf.status:
        eq = sf.EQU(shot, diag='EQH')

    rho_trz = RegularGridInterpolator(
        points=(
            np.array(eq.Rmesh, dtype=np.float32),
            np.array(eq.Zmesh, dtype=np.float32),
            np.array(eq.time, dtype=np.float32),
        ),
        values=np.array(np.sqrt((eq.pfm - eq.psi0) / (eq.psix - eq.psi0)), dtype=np.float32),
        bounds_error=False,
        fill_value=np.nan,
    )

    return rho_trz


def r_to_rho(t, r, shot, side):
    # TODO: Get these values from the rpspy library
    zlfs = 0.14  # Antenna height on LFS
    zhfs = 0.07  # Antenna height on HFS

    if side.lower() == 'lfs':
        z = zlfs
    elif side.lower() == 'hfs':
        z = zhfs
    else:
        raise ValueError("Invalid side. Choose 'lfs' or 'hfs'.")

    if AUG_MODE:
        [t, r, z] = np.broadcast_arrays(t, r, z)
        interpolator = cached_get_equilibrium_interpolator(shot)
        rho = interpolator(np.array([r.flatten(), z.flatten(), t.flatten()], dtype=np.float32).T).reshape(r.shape)
    else:
        print("ipfnpytools and aug_sfutils not found. Cannot calculate rho.")
        rho = r * 0.0

    return rho


def round_to_nearest(value, value_list):
    return min(value_list, key=lambda x: abs(x - value))


def get_shot_from_path(path):
    return int(os.path.basename(path))


def get_path_from_shot(shot):
    return f"/shares/experiments/aug-rawfiles/RTR/{shot // 10}/{shot}"


@lru_cache(maxsize=500)
def cached_get_auto_linearization_from_shares(shot, band, sweep):
    return rpspy.get_linearization(shot, sweep, band, use_lookup_table=True)


@lru_cache(maxsize=100)
def cached_full_profile_reconstruction(*args, **kwargs):
    kwargs['spectrogram_options'] = json.loads(kwargs['spectrogram_options'])
    kwargs['filters'] = json.loads(kwargs['filters'])
    return rpspy.full_profile_reconstruction(*args, **kwargs)
