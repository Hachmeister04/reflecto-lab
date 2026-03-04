import logging
import json
import numpy as np
from scipy.signal import spectrogram
from scipy.fft import fftshift
import scipy.constants as cst
import rpspy

from constants import (
    BANDS, SIDES,
    DEFAULT_LINEARIZATION_SWEEP,
    PROFILE_INVERSION_RESOLUTION,
)
from model.state import (
    SpectrogramParams, FilterRange, ExclusionRange,
    InitValues, InitFileData, BeatFrequencyData,
    CurrentSweepData, CurrentFFTData, CurrentDisplayData,
    AggregatedDelayData, DetectorSelection,
)
from utils.helpers import cached_get_auto_linearization_from_shares, r_to_rho

logger = logging.getLogger(__name__)


class ShotModel:
    """Pure data model for reflectometry analysis. No PyQt dependency."""

    def __init__(self):
        # Shot identity
        self.shot = 0
        self.file_path = ''
        self.time_stamps = None

        # Per-detector state: Dict[side][band]
        self.spect_params = {
            side: {
                band: SpectrogramParams(
                    subtract_dispersion=True if band == 'V' else None
                ) for band in BANDS
            } for side in SIDES
        }
        self.filters = {
            side: {band: FilterRange() for band in BANDS} for side in SIDES
        }
        self.beat_frequencies = {
            side: {band: BeatFrequencyData() for band in BANDS} for side in SIDES
        }
        self.background_spectrograms = {
            side: {band: None for band in BANDS} for side in SIDES
        }

        # Per-side state
        self.exclusion_filters = {side: [] for side in SIDES}
        self.init_values = {side: InitValues() for side in SIDES}

        # Current detector selection
        self.detector = DetectorSelection()

        # Current computed data
        self.current_sweep = CurrentSweepData()
        self.current_fft = CurrentFFTData()
        self.current_display = CurrentDisplayData()

        # Aggregated delay data
        self.aggregated_hfs = AggregatedDelayData()
        self.aggregated_lfs = AggregatedDelayData()

        # Limiter/initialization
        self.inner_limiter = 0.0
        self.outer_limiter = 0.0
        self.hfs_gd_at_zero_fp = 0.0
        self.lfs_gd_at_zero_fp = 0.0
        self.get_init = None

    # --- Shot loading ---

    def load_shot_from_path(self, file_path):
        """Load shot from file path. Raises FileNotFoundError/ValueError on failure."""
        from utils.helpers import get_shot_from_path
        shot = get_shot_from_path(file_path)
        rpspy.get_timestamps(shot, file_path)
        self.shot = shot
        self.file_path = file_path

    def load_shot_from_number(self, shot):
        """Load shot by number. Raises FileNotFoundError on failure."""
        from utils.helpers import get_path_from_shot
        file_path = get_path_from_shot(shot)
        rpspy.get_timestamps(shot, file_path)
        self.shot = shot
        self.file_path = file_path

    def post_load_init(self):
        """Run after successful shot load: backgrounds, limiters, timestamps."""
        self.compute_all_backgrounds()
        self._init_default_limiters()
        self.time_stamps = rpspy.get_timestamps(self.shot, self.file_path)

    def _init_default_limiters(self):
        """Set default limiter positions from rpspy."""
        self.inner_limiter = rpspy.get_reflectometry_limiter_position('hfs')
        self.hfs_gd_at_zero_fp = 2 * (self.inner_limiter - rpspy.get_antenna_position('hfs')) / cst.c
        self.init_values['HFS'].current_value = self.inner_limiter

        self.outer_limiter = rpspy.get_reflectometry_limiter_position('lfs')
        self.lfs_gd_at_zero_fp = 2 * (rpspy.get_antenna_position('lfs') - self.outer_limiter) / cst.c
        self.init_values['LFS'].current_value = self.outer_limiter

    # --- Initialization ---

    def initialize_limiters(self, side, timestamp, update_all_sides=True, define_init_funcs=False):
        """Compute limiter positions based on init_values.

        Args:
            side: Current active side ('HFS' or 'LFS').
            timestamp: Current sweep timestamp.
            update_all_sides: If True, also process 'From file' init for both sides.
            define_init_funcs: If True, create self.get_init closure.
        """
        iv = self.init_values[side]

        if iv.type == 'Default (recommended)':
            if side == 'HFS':
                self.inner_limiter = rpspy.get_reflectometry_limiter_position('hfs')
                self.hfs_gd_at_zero_fp = 2 * (self.inner_limiter - rpspy.get_antenna_position('hfs')) / cst.c
            elif side == 'LFS':
                self.outer_limiter = rpspy.get_reflectometry_limiter_position('lfs')
                self.lfs_gd_at_zero_fp = 2 * (rpspy.get_antenna_position('lfs') - self.outer_limiter) / cst.c

        elif iv.type == 'Custom':
            if side == 'HFS':
                self.inner_limiter = iv.custom_value
                self.hfs_gd_at_zero_fp = 2 * (self.inner_limiter - rpspy.get_antenna_position('hfs')) / cst.c
            elif side == 'LFS':
                self.outer_limiter = iv.custom_value
                self.lfs_gd_at_zero_fp = 2 * (rpspy.get_antenna_position('lfs') - self.outer_limiter) / cst.c

        if update_all_sides:
            for s in SIDES:
                ivs = self.init_values[s]
                if ivs.type == 'From file' and ivs.file.name != '':
                    limiter = np.interp(timestamp, ivs.file.time, ivs.file.position)
                    if s == 'HFS':
                        self.inner_limiter = limiter
                        self.hfs_gd_at_zero_fp = 2 * (self.inner_limiter - rpspy.get_antenna_position('hfs')) / cst.c
                        ivs.current_value = self.inner_limiter
                    if s == 'LFS':
                        self.outer_limiter = limiter
                        self.lfs_gd_at_zero_fp = 2 * (rpspy.get_antenna_position('lfs') - self.outer_limiter) / cst.c
                        ivs.current_value = self.outer_limiter

        if side == 'HFS':
            self.init_values[side].current_value = self.inner_limiter
        elif side == 'LFS':
            self.init_values[side].current_value = self.outer_limiter

        if define_init_funcs:
            def get_init(s, time):
                if self.init_values[s].type == 'From file':
                    return np.interp(time, self.init_values[s].file.time, self.init_values[s].file.position)
                else:
                    return self.inner_limiter if s == 'HFS' else self.outer_limiter
            self.get_init = get_init

    def get_current_limiter_value(self, side):
        """Get the current limiter value for the given side."""
        if side == 'HFS':
            return self.inner_limiter
        return self.outer_limiter

    # --- Sweep computation ---

    def compute_sweep(self):
        """Compute linearized sweep data for current detector."""
        d = self.detector
        signal_type = 'complex' if d.band == 'V' else 'real'

        data = rpspy.get_band_signal(
            self.shot, self.file_path, d.band, d.side, signal_type, d.sweep
        )[0]
        data -= 2**11 if signal_type == 'real' else (2**11 + 1j * 2**11)

        x_data = cached_get_auto_linearization_from_shares(self.shot, d.band, d.sweep)

        logger.debug("sweep: %d", d.sweep)

        x_data, data = rpspy.linearize(x_data, data)

        self.current_sweep = CurrentSweepData(
            data=data,
            x_data=x_data,
            signal_type=signal_type,
        )

    # --- Spectrogram computation ---

    def compute_spectrogram(self, band, side, nperseg, noverlap, nfft, sweep, burst_size, subtract_dispersion):
        """Calculate spectrogram for a given band and side.

        Returns:
            tuple: (f, fs, f_beat, t, f_probe, Sxx)
        """
        signal_type = 'complex' if band == 'V' else 'real'

        burst = rpspy.get_band_signal(
            self.shot, self.file_path, band, side, signal_type, sweep - burst_size // 2, burst_size
        )

        f = cached_get_auto_linearization_from_shares(self.shot, band, sweep)

        f, linearized_burst = rpspy.linearize(f, burst)

        fs = rpspy.get_sampling_frequency(self.shot, self.file_path)

        if band == 'V':
            linearized_burst -= (2**11 + 1j * 2**11)
        else:
            linearized_burst -= 2**11

        if subtract_dispersion is True:
            df_dt = (f[-1] - f[0]) / ((len(f) - 1) * (1 / fs))
            correction = rpspy.get_dispersion_phase(self.shot, band, side, f[0], df_dt, np.arange(len(f)) / fs)
            corrected_burst = linearized_burst * correction
        else:
            corrected_burst = linearized_burst

        f_beat, t, Sxx = spectrogram(
            corrected_burst,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            return_onesided=False if corrected_burst.dtype == complex else True,
            detrend=False,
            window='boxcar',
        )

        f_probe = np.interp(t, np.arange(len(f)) / fs, f)

        if band == 'V':
            f_beat = fftshift(f_beat)
            Sxx = fftshift(Sxx, axes=-2)

        # Average across burst
        Sxx = np.average(Sxx, axis=0)

        return f, fs, f_beat, t, f_probe, Sxx

    def compute_current_fft(self):
        """Compute full FFT pipeline for current detector selection."""
        d = self.detector
        sp = self.spect_params[d.side][d.band]
        f, fs, f_beat, t, f_probe, Sxx = self.compute_spectrogram(
            d.band, d.side, sp.nperseg, sp.noverlap, sp.nfft,
            d.sweep, d.burst_size, sp.subtract_dispersion
        )
        self.current_fft = CurrentFFTData(
            f=f, fs=fs, f_beat=f_beat, t=t, f_probe=f_probe, unfiltered_Sxx=Sxx,
        )

    # --- Dispersion ---

    def compute_dispersion(self, band, side, f_probe, t, subtract):
        """Compute dispersion for the selected band and side."""
        if band == 'V' and subtract is True:
            return np.zeros(len(f_probe))
        else:
            k = (f_probe[-1] - f_probe[0]) / (t[-1] - t[0])
            return k * rpspy.aug_tgcorr2(band, side, f_probe * 1e-9, self.shot)

    # --- Beat frequency ---

    def compute_beatf(self, band, side, Sxx, y_dis, f_beat, fs):
        """Compute beat frequency from spectrogram data."""
        filt = self.filters[side][band]

        Sxx[np.broadcast_to(f_beat[:, None], Sxx.shape) <= y_dis + filt.low] = Sxx.min()
        Sxx[np.broadcast_to(f_beat[:, None], Sxx.shape) >= y_dis + filt.high] = Sxx.min()

        y_max, _ = rpspy.column_wise_max_with_quadratic_interpolation(Sxx)
        y_max *= abs(f_beat[1] - f_beat[0])

        if band == 'V':
            y_max += -fs / 2

        return y_max

    def compute_one_beatf(self, band, side):
        """Update beat freq data for one band/side."""
        d = self.detector
        if side == d.side and band == d.band:
            fft = self.current_fft
            disp = self.current_display
            df_dt = (fft.f[-1] - fft.f[0]) / ((len(fft.f) - 1) * (1 / fft.fs))
            y_beat_time = (disp.y_beatf - disp.y_dis) / df_dt
            self.beat_frequencies[side][band] = BeatFrequencyData(
                f_probe=fft.f_probe, y_beatf=disp.y_beatf,
                y_beat_time=y_beat_time, df_dt=df_dt,
            )
        else:
            sp = self.spect_params[side][band]
            f, fs, f_beat, t, f_probe, Sxx = self.compute_spectrogram(
                band, side, sp.nperseg, sp.noverlap, sp.nfft,
                d.sweep, d.burst_size, sp.subtract_dispersion,
            )

            if sp.subtract_background:
                Sxx = self.background_subtract(Sxx, band, side)

            y_dis = self.compute_dispersion(band, side, f_probe, t, sp.subtract_dispersion)
            y_beatf = self.compute_beatf(band, side, Sxx, y_dis, f_beat, fs)

            df_dt = (f[-1] - f[0]) / ((len(f) - 1) * (1 / fs))
            y_beat_time = (y_beatf - y_dis) / df_dt

            self.beat_frequencies[side][band] = BeatFrequencyData(
                f_probe=f_probe, y_beatf=y_beatf,
                y_beat_time=y_beat_time, df_dt=df_dt,
            )

    def compute_all_beatf(self):
        """Update beat frequencies for all 8 detectors."""
        for side in SIDES:
            for band in BANDS:
                self.compute_one_beatf(band, side)

    # --- Background ---

    def compute_background(self, band, side):
        """Calculate and store background spectrogram."""
        sp = self.spect_params[side][band]
        burst_size = self.detector.burst_size
        sweep = burst_size // 2

        _, _, _, _, _, Sxx = self.compute_spectrogram(
            band, side, sp.nperseg, sp.noverlap, sp.nfft,
            sweep, burst_size, sp.subtract_dispersion,
        )
        self.background_spectrograms[side][band] = Sxx

    def compute_all_backgrounds(self):
        """Calculate backgrounds for all band/side combinations."""
        for side in SIDES:
            for band in BANDS:
                self.compute_background(band, side)

    def background_subtract(self, Sxx, band, side):
        """Subtract background from spectrogram."""
        return np.clip(
            Sxx - self.background_spectrograms[side][band],
            a_min=np.min(Sxx),
            a_max=np.inf,
        )

    # --- Aggregated delays ---

    def compute_aggregated_delays(self):
        """Build aggregated delay arrays for HFS and LFS, applying exclusions."""
        for agg_data, side, gd_at_zero in [
            (self.aggregated_hfs, 'HFS', self.hfs_gd_at_zero_fp),
            (self.aggregated_lfs, 'LFS', self.lfs_gd_at_zero_fp),
        ]:
            all_f_probe = np.concatenate((
                [0],
                self.beat_frequencies[side]['K'].f_probe,
                self.beat_frequencies[side]['Ka'].f_probe,
                self.beat_frequencies[side]['Q'].f_probe,
                self.beat_frequencies[side]['V'].f_probe,
            ))
            all_beat_time = np.concatenate((
                [gd_at_zero],
                self.beat_frequencies[side]['K'].y_beat_time,
                self.beat_frequencies[side]['Ka'].y_beat_time,
                self.beat_frequencies[side]['Q'].y_beat_time,
                self.beat_frequencies[side]['V'].y_beat_time,
            ))

            # Remove NaNs
            nan_mask = np.isnan(all_beat_time)
            all_beat_time = all_beat_time[~nan_mask]
            all_f_probe = all_f_probe[~nan_mask]

            # Apply exclusion filters
            for excl in self.exclusion_filters[side]:
                mask = (all_f_probe >= excl.low) & (all_f_probe <= excl.high) & (all_f_probe != 0)
                all_f_probe = all_f_probe[~mask]
                all_beat_time = all_beat_time[~mask]

            agg_data.f_probe = all_f_probe
            agg_data.beat_time = all_beat_time

    # --- Profile ---

    def compute_profile(self, coordinate_mode, timestamp):
        """Compute density profile.

        Returns:
            tuple: (r_HFS, ne_HFS, r_LFS, ne_LFS)
        """
        # HFS
        gd_hfs_x = np.linspace(
            self.aggregated_hfs.f_probe[0], self.aggregated_hfs.f_probe[-1],
            PROFILE_INVERSION_RESOLUTION,
        )
        gd_hfs_y = np.interp(gd_hfs_x, self.aggregated_hfs.f_probe, self.aggregated_hfs.beat_time)

        # LFS
        gd_lfs_x = np.linspace(
            self.aggregated_lfs.f_probe[0], self.aggregated_lfs.f_probe[-1],
            PROFILE_INVERSION_RESOLUTION,
        )
        gd_lfs_y = np.interp(gd_lfs_x, self.aggregated_lfs.f_probe, self.aggregated_lfs.beat_time)

        r_HFS = rpspy.profile_inversion(
            gd_hfs_x,
            np.clip(gd_hfs_y - gd_hfs_y[0], a_min=-np.inf, a_max=np.inf),
            pwld_batch=True,
        )
        r_LFS = rpspy.profile_inversion(
            gd_lfs_x,
            np.clip(gd_lfs_y - gd_lfs_y[0], a_min=-np.inf, a_max=np.inf),
            pwld_batch=True,
        )

        r_HFS = r_HFS + self.inner_limiter
        r_LFS = -r_LFS + self.outer_limiter

        ne_HFS = rpspy.f_to_ne(gd_hfs_x)
        ne_LFS = rpspy.f_to_ne(gd_lfs_x)

        if coordinate_mode == 'rho-poloidal':
            r_HFS = r_to_rho(timestamp, r_HFS, self.shot, 'HFS')
            r_LFS = r_to_rho(timestamp, r_LFS, self.shot, 'LFS')

        return r_HFS, ne_HFS, r_LFS, ne_LFS

    # --- Display data (for current spectrogram view) ---

    def compute_current_display(self):
        """Compute dispersion and beat frequency for the current detector view."""
        d = self.detector
        fft = self.current_fft
        sp = self.spect_params[d.side][d.band]

        # Apply background subtraction
        if sp.subtract_background:
            Sxx = self.background_subtract(np.array(fft.unfiltered_Sxx), d.band, d.side)
        else:
            Sxx = np.array(fft.unfiltered_Sxx)
        self.current_fft.Sxx = Sxx

        y_dis = self.compute_dispersion(d.band, d.side, fft.f_probe, fft.t, sp.subtract_dispersion)
        y_beatf = self.compute_beatf(d.band, d.side, np.array(Sxx), y_dis, fft.f_beat, fft.fs)

        self.current_display = CurrentDisplayData(
            y_beatf=y_beatf,
            y_dis=y_dis,
            df_dt=(fft.f[-1] - fft.f[0]) / ((len(fft.f) - 1) * (1 / fft.fs)),
        )

    # --- Config serialization ---

    def save_config(self, path):
        """Serialize current params/filters/exclusions to JSON."""
        params_dict = {}
        for side in SIDES:
            params_dict[side] = {}
            for band in BANDS:
                params_dict[side][band] = self.spect_params[side][band].to_config_dict()

        filters_dict = {}
        for side in SIDES:
            filters_dict[side] = {}
            for band in BANDS:
                filters_dict[side][band] = self.filters[side][band].to_config_list()

        exclusions_dict = {}
        for side in SIDES:
            exclusions_dict[side] = [excl.to_config_list() for excl in self.exclusion_filters[side]]

        data = {
            'parameters': params_dict,
            'filters': filters_dict,
            'burst_size': self.detector.burst_size,
            'exclusion_filters': exclusions_dict,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

        logger.info("Config saved to %s", path)

    def load_config(self, path):
        """Deserialize params/filters/exclusions from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        params = data.get('parameters', {})
        for side in SIDES:
            for band in BANDS:
                if side in params and band in params[side]:
                    self.spect_params[side][band] = SpectrogramParams.from_config_dict(params[side][band])

        filters = data.get('filters', {})
        for side in SIDES:
            for band in BANDS:
                if side in filters and band in filters[side]:
                    self.filters[side][band] = FilterRange.from_config_list(filters[side][band])

        self.detector.burst_size = data.get('burst_size', 1)

        exclusions = data.get('exclusion_filters', {})
        for side in SIDES:
            if side in exclusions:
                self.exclusion_filters[side] = [
                    ExclusionRange.from_config_list(e) for e in exclusions[side]
                ]
