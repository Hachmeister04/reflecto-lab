import os
import rpspy
import json
import numpy as np
from functools import lru_cache
from PyQt5.QtWidgets import QMessageBox
from scipy.interpolate import RegularGridInterpolator

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
    rho_trz: ndarray
        Rho poloidal coordinates.
        
    """
    
    eq = sf.EQU(shot, diag='IDE')
    if not eq.sf.status:
        eq = sf.EQU(shot, diag='EQH')

    print(eq)
    print(eq.__dict__)
    
    rho_trz = RegularGridInterpolator(
        points=(
            np.array(eq.Rmesh, dtype=np.float32), 
            np.array(eq.Zmesh, dtype=np.float32),
            np.array(eq.time, dtype=np.float32),
        ), 
        values=np.array(np.sqrt((eq.pfm - eq.psi0)/(eq.psix-eq.psi0)), dtype=np.float32), 
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

def round_to_nearest(value: float, value_list: list):
    return min(value_list, key=lambda x: abs(x - value))


def get_shot_from_path(path: str):
    return int(os.path.basename(path))


def get_path_from_shot(shot: int):
    return f"/shares/experiments/aug-rawfiles/RTR/{shot//10}/{shot}"


def show_warning():
    # Create a warning message box
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Warning")
    msg.setText("Raw data not found for the selected shot \n OR \n Selected folder does not contain valid raw data.")
    #msg.setInformativeText("Selected folder does not contain valid raw data.")
    msg.setStandardButtons(QMessageBox.Ok)
    #msg.setDefaultButton(QMessageBox.Ok)
    
    # Show the message box
    msg.exec()


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


def _get_signal_on_window(shot, shotfile, signal, window='all'):
    """Internal function used by `get_average` and `get_percentiles` to load a timme signal on a given window"""
    
    from ipfnpytools.getsig import getsig, gettime
    from ipfnpytools.current_flattop import current_flattop

    try:
        data = getsig(shot, shotfile, signal)
        
        if data is None:
            return None

        if window == 'all':
            return gettime(shot, shotfile, signal), np.average(data)

        else:
            time = gettime(shot, shotfile, signal)
            
            if time is None:
                return None

            if window == 'flattop':
                try:
                    t0, t1 = current_flattop(shot)
                except ValueError:
                    return None
                except AttributeError:
                    return None
            else:
                t0, t1 = window

            mask = (time > t0) & (time < t1)
            return time[mask], data[mask]
    except AttributeError:
        return None

def get_average(shot, shotfile, signal, window='all'):
    """Compute the average of an AUG signal.
    
    Parameters
    ----------
    shot: int
        Shot number
    shotfile: str
        Three-letter shotfile identifier
    signal: str
        Signal name to compute the average
    window: {'all', 'flatttop'} or 2-float tuple
        Window to compute the signal average. 
        Choose 'all' for the entire signal;
        or 'flattop' for automatic current flattop detection;
        or provide a 2-valued tuple to manually set the window.
        
    Returns
    -------
    average: float
        Signal average on the specified time window.
    """
    
    _, data = _get_signal_on_window(shot=shot, shotfile=shotfile, signal=signal, window=window)
    if data is None:
        return None
    else:
        return np.nanmean(data)