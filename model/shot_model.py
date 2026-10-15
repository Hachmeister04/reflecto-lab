import logging
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import scipy.constants as cst
import rpspy

from constants import (
    BANDS, SIDES,
    DEFAULT_LINEARIZATION_SWEEP,
    PROFILE_INVERSION_RESOLUTION,
    DEFAULT_START_TIME, DEFAULT_END_TIME, DEFAULT_TIMESTEP, 
    DEFAULT_CUSTOM_DENSITY_CUTOFF, DEFAULT_DENSITY_CUTOFF, DEFAULT_CUSTOM_DENSITY_CUTOFF_MULTIPLIER
)
from model.state import (
    SpectrogramParams, FilterRange, FrequencyExclusion, SpectrogramMask,
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
                    subtract_dispersion=True if band == 'V' else False
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

        # Per-side 1D probing-frequency intervals dropped from the merged profile
        self.frequency_exclusions = {side: [] for side in SIDES}
        self.init_values = {side: InitValues() for side in SIDES}

        # Per-detector (band+side) 2D spectrogram boxes blanked before peak-finding
        self.spectrogram_masks = {side: {band: [] for band in BANDS} for side in SIDES}

        # Current detector selection
        self.detector = DetectorSelection()

        # Reconstruction time range
        self.reconstruction_start_time = DEFAULT_START_TIME
        self.reconstruction_end_time = DEFAULT_END_TIME
        self.reconstruction_time_step = DEFAULT_TIMESTEP

        # Current computed data
        self.current_sweep = CurrentSweepData()
        self.current_fft = CurrentFFTData()
        self.current_display = CurrentDisplayData()
        self.current_band = None  # rpspy.Band of the current detector, see compute_band_burst

        # Aggregated delay data
        self.aggregated_hfs = AggregatedDelayData()
        self.aggregated_lfs = AggregatedDelayData()

        # Cached shot-level data
        self.sampling_frequency = None

        # Persistent thread pool for parallel computation
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Limiter/initialization
        self.inner_limiter = 0.0
        self.outer_limiter = 0.0
        self.hfs_gd_at_zero_fp = 0.0
        self.lfs_gd_at_zero_fp = 0.0
        self.get_init = None

        # Custom density cutoff
        self.custom_density_cutoff = DEFAULT_CUSTOM_DENSITY_CUTOFF 
        self.custom_density_cutoff_value = DEFAULT_DENSITY_CUTOFF
        self.custom_density_cutoff_multiplier = DEFAULT_CUSTOM_DENSITY_CUTOFF_MULTIPLIER
        #self.density_cutoff_times = None
        #self.density_cutoff_values = None

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
        self.sampling_frequency = rpspy.get_sampling_frequency(self.shot, self.file_path)
        self.time_stamps = rpspy.get_timestamps(self.shot, self.file_path)
        self.prewarm_shot_wide_data()
        self.compute_all_backgrounds()
        self._init_default_limiters()

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

    # --- Processing ---
    #
    # Everything below calls the same rpspy functions that rpspy.full_profile_reconstruction
    # runs, on the same settings, for a single instant. Do not reimplement a processing step
    # here: the preview would silently stop matching the reconstruction.

    @staticmethod
    def signal_type(band):
        return 'complex' if band == 'V' else 'real'

    def _background_burst_size(self, sp):
        """Burst size of the background, which defaults to the burst size of the signal."""
        return sp.background_burst_size if sp.background_burst_size is not None else self.detector.burst_size

    def _current_time(self):
        """Time of the sweep being looked at, as the single instant rpspy processes."""
        return np.array([self.time_stamps[self.detector.sweep]])

    def prewarm_shot_wide_data(self):
        """Read, in parallel, what rpspy needs from the whole shot and caches.

        The probing-frequency axis of each band comes from the linearization tables of the
        whole shot, and the noise filter of the HFS profile from the whole Q-HFS signal. The
        first request of each takes a while, which is better spent at load than at the first plot.
        """
        futures = [
            self._executor.submit(rpspy.get_common_frequency_axis, self.shot, band)
            for band in BANDS
        ]
        futures.append(self._executor.submit(rpspy.get_qhfs_signal_deviation, self.shot, self.file_path))
        for f in futures:
            f.result()

    # --- Sweep computation ---

    def compute_sweep(self):
        """Compute linearized sweep data for current detector."""
        import time as _time
        d = self.detector
        signal_type = self.signal_type(d.band)

        _t0 = _time.perf_counter()
        data = rpspy.get_band_signal(
            self.shot, self.file_path, d.band, d.side, signal_type, d.sweep
        )[0]
        data -= 2**11 if signal_type == 'real' else (2**11 + 1j * 2**11)
        _t1 = _time.perf_counter()

        table = cached_get_auto_linearization_from_shares(self.shot, d.band, d.sweep)
        _t2 = _time.perf_counter()

        x_data, data = rpspy.linearize(table, data, new_x=rpspy.get_common_frequency_axis(self.shot, d.band))
        _t3 = _time.perf_counter()

        logger.info(
            "  compute_sweep: get_band_signal=%.1fms linearization_cache=%.1fms linearize=%.1fms",
            (_t1 - _t0) * 1000, (_t2 - _t1) * 1000, (_t3 - _t2) * 1000,
        )

        self.current_sweep = CurrentSweepData(
            data=data,
            x_data=x_data,
            signal_type=signal_type,
        )

    # --- Spectrogram computation ---

    def compute_band_burst(self, band, side, sweep, burst_size, sp):
        """Burst-averaged spectrogram of a band and side around `sweep`.

        Returns:
            rpspy.Band: with `burst_spectrogram` of shape (1, n_beat, n_probe), the axes
            `x` (probing frequency) and `y` (beat frequency), `frequency` (linearized sweep axis),
            `sweep_rate`, `dispersion` and `tau_offset`.
        """
        all_sweeps, burst_mode = rpspy.burst_windows(sweep, burst_size)
        return rpspy.band_burst_spectrogram(
            self.shot, band, side, self.signal_type(band), all_sweeps, burst_mode,
            subtract_dispersion=bool(sp.subtract_dispersion),
            nperseg=sp.nperseg, noverlap=sp.noverlap, nfft=sp.nfft,
            shotfile_dir=self.file_path, shot_linearization=self.shot,
            linearization_shotfile_dir=self.file_path,
        )

    def compute_group_delay(self, band_data, band, side, spectrogram, copy):
        """Group delay of one band and side at the current sweep.

        Args:
            band_data: rpspy.Band from compute_band_burst.
            spectrogram: (1, n_beat, n_probe) spectrogram, already background-subtracted if desired.
            copy: Leave `spectrogram` untouched.

        Returns:
            tuple: (group delay, beat frequency), each of shape (1, n_probe).
        """
        filt = self.filters[side][band]
        masks = [m.to_rpspy() for m in self.spectrogram_masks[side][band] if m.enabled]
        return rpspy.find_group_delay(
            spectrogram, band_data, filt.low, filt.high,
            masks=masks, times=self._current_time(), copy=copy,
        )

    def _spectrogram_for_group_delay(self, band_data, band, side):
        """The spectrogram the group delay is found on: background-subtracted if enabled."""
        if self.spect_params[side][band].subtract_background:
            return rpspy.subtract_background(band_data.burst_spectrogram, self.background_spectrograms[side][band])
        return band_data.burst_spectrogram

    def compute_current_fft(self):
        """Compute the spectrogram for current detector selection."""
        d = self.detector
        sp = self.spect_params[d.side][d.band]
        band_data = self.compute_band_burst(d.band, d.side, d.sweep, d.burst_size, sp)
        self.current_band = band_data
        self.current_fft = CurrentFFTData(
            f=band_data.frequency, fs=band_data.sampling_frequency, f_beat=band_data.y,
            f_probe=band_data.x, unfiltered_Sxx=band_data.burst_spectrogram[0],
        )

    # --- Beat frequency ---

    def compute_one_beatf(self, band, side):
        """Update beat freq data for one band/side."""
        d = self.detector
        if side == d.side and band == d.band:
            fft = self.current_fft
            disp = self.current_display
            self.beat_frequencies[side][band] = BeatFrequencyData(
                f_probe=fft.f_probe, y_beatf=disp.y_beatf,
                y_beat_time=disp.y_beat_time, df_dt=disp.df_dt,
            )
        else:
            self.beat_frequencies[side][band] = self._compute_one_beatf_result(band, side)

    def _compute_one_beatf_result(self, band, side):
        """Compute beat freq for one non-current band/side. Returns result (thread-safe)."""
        d = self.detector
        sp = self.spect_params[side][band]
        band_data = self.compute_band_burst(band, side, d.sweep, d.burst_size, sp)
        spectrogram = self._spectrogram_for_group_delay(band_data, band, side)
        tau, beat_frequency = self.compute_group_delay(band_data, band, side, spectrogram, copy=False)

        return BeatFrequencyData(
            f_probe=band_data.x, y_beatf=beat_frequency[0],
            y_beat_time=tau[0], df_dt=band_data.sweep_rate,
        )

    def compute_all_beatf(self):
        """Update beat frequencies for all 8 detectors in parallel."""
        import time as _time
        _t_start = _time.perf_counter()
        d = self.detector

        # Current detector — instant, uses cached FFT/display data
        self.compute_one_beatf(d.band, d.side)

        # Remaining 7 detectors — run in parallel
        others = [
            (band, side) for side in SIDES for band in BANDS
            if not (side == d.side and band == d.band)
        ]

        futures = {
            self._executor.submit(self._compute_one_beatf_result, band, side): (band, side)
            for band, side in others
        }
        for future in futures:
            band, side = futures[future]
            self.beat_frequencies[side][band] = future.result()

        _t_end = _time.perf_counter()
        logger.info("  all_beatf: %.1fms (7 detectors parallel)", (_t_end - _t_start) * 1000)

    # --- Background ---

    def compute_background(self, band, side):
        """Calculate and store background spectrogram."""
        sp = self.spect_params[side][band]
        self.background_spectrograms[side][band] = rpspy.background_spectrogram(
            self.shot, band, side, self.signal_type(band),
            sp.background_sweep, self._background_burst_size(sp),
            subtract_dispersion=bool(sp.subtract_dispersion),
            nperseg=sp.nperseg, noverlap=sp.noverlap, nfft=sp.nfft,
            shotfile_dir=self.file_path, shot_linearization=self.shot,
            linearization_shotfile_dir=self.file_path,
            n_sweeps=len(self.time_stamps),
        )

    def compute_all_backgrounds(self):
        """Calculate backgrounds for all band/side combinations."""
        for side in SIDES:
            for band in BANDS:
                self.compute_background(band, side)

    # --- Profile ---

    def _limiter(self, side, time):
        """Initialization position at `time`, as rpspy.full_profile_reconstruction asks for it."""
        if self.get_init is not None:
            return self.get_init(side, time)
        return self.inner_limiter if side == 'HFS' else self.outer_limiter

    def _invert_current(self):
        """Merge the group delay of the 8 detectors and invert it, as the reconstruction does."""
        t = self._current_time()
        bands = {
            side: [
                (self.beat_frequencies[side][band].f_probe, self.beat_frequencies[side][band].y_beat_time[None, :])
                for band in BANDS
            ] for side in SIDES
        }
        exclusions = {
            side: [e.to_rpspy() for e in self.frequency_exclusions[side] if e.enabled] for side in SIDES
        }
        return rpspy.reconstruct_profiles(
            t, bands['HFS'], bands['LFS'], exclusions,
            inner_limiter=self._limiter('HFS', t), outer_limiter=self._limiter('LFS', t),
        )

    def compute_aggregated_delays(self, timestamp):
        """Build aggregated delay arrays for HFS and LFS: the points the profile is built from."""
        profiles = self._invert_current()
        for agg_data, fp, gd in [
            (self.aggregated_hfs, profiles.fp_hfs, profiles.gd_hfs[0]),
            (self.aggregated_lfs, profiles.fp_lfs, profiles.gd_lfs[0]),
        ]:
            valid = ~np.isnan(gd)
            agg_data.f_probe = fp[valid]
            agg_data.beat_time = gd[valid]

    def compute_profile(self, coordinate_mode, timestamp):
        """Compute density profile.

        The profile is what the reconstruction produces for this instant, before the density
        cutoff: the points above the cutoff are returned too so they can be shown differently.

        Returns:
            tuple: (r_HFS, ne_HFS, r_LFS, ne_LFS)
        """
        profiles = self._invert_current()

        # Same discard of noisy HFS profiles as in the reconstruction
        rpspy.apply_qhfs_noise_filter(
            self.shot, self.file_path, profiles.times, profiles.f, profiles.r_hfs, profiles.gds_hfs,
        )

        out = []
        for r, side in ((profiles.r_hfs[0], 'HFS'), (profiles.r_lfs[0], 'LFS')):
            valid = np.isfinite(r)
            r, ne = r[valid], profiles.ne[valid]
            if coordinate_mode == 'rho-poloidal':
                r = r_to_rho(timestamp, r, self.shot, side)
            out += [r, ne]

        return tuple(out)

    # --- Display data (for current spectrogram view) ---

    def compute_current_display(self):
        """Compute dispersion and beat frequency for the current detector view."""
        d = self.detector
        band_data = self.current_band
        sp = self.spect_params[d.side][d.band]

        spectrogram = self._spectrogram_for_group_delay(band_data, d.band, d.side)
        tau, beat_frequency = self.compute_group_delay(band_data, d.band, d.side, spectrogram, copy=True)

        # What is drawn: where the background is larger than the signal, show the lowest signal
        # instead, so that a logarithmic scale can be drawn. Not used to find the group delay.
        shown = spectrogram[0]
        if sp.subtract_background:
            shown = np.clip(shown, a_min=np.min(band_data.burst_spectrogram), a_max=np.inf)
        self.current_fft.Sxx = shown

        self.current_display = CurrentDisplayData(
            y_beatf=beat_frequency[0],
            y_dis=band_data.dispersion,
            y_beat_time=tau[0],
            df_dt=band_data.sweep_rate,
        )

    # --- Reconstruction time grid ---

    def reconstruction_sweeps(self):
        """Sweeps at which a reconstruction with the current settings is computed."""
        _, central_sweeps, _ = rpspy.burst_mode_handler(
            self.time_stamps, self.reconstruction_start_time, self.reconstruction_end_time,
            self.reconstruction_time_step, self.detector.burst_size,
        )
        return central_sweeps

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
            exclusions_dict[side] = [
                excl.to_config_list() for excl in self.frequency_exclusions[side] if excl.enabled
            ]

        masks_dict = {}
        for side in SIDES:
            masks_dict[side] = {}
            for band in BANDS:
                masks_dict[side][band] = [
                    mask.to_config_list() for mask in self.spectrogram_masks[side][band]
                ]


        data = {
            'parameters': params_dict,
            'filters': filters_dict,
            'burst_size': self.detector.burst_size,
            'frequency_exclusions': exclusions_dict,
            'exclusion_filters': exclusions_dict,  # for backward compatibility
            'spectrogram_masks': masks_dict,
            'exclusion_regions': masks_dict,  # for backward compatibility
            'reconstruction_times': {
                'start_time': self.reconstruction_start_time,
                'end_time': self.reconstruction_end_time,
                'time_step': self.reconstruction_time_step,
            },
            'custom_density_cutoff': {
                'custom_density_cutoff': self.custom_density_cutoff, #TODO: best to save the array of times and densities??
                'custom_density_cutoff_value': self.custom_density_cutoff_value,
                'custom_density_cutoff_multiplier': self.custom_density_cutoff_multiplier,
            },
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

        logger.info("Config saved to %s", path)

    def load_config(self, path):
        """Deserialize params/filters/exclusions from JSON."""
        print("before load:", self.frequency_exclusions)
        with open(path, 'r') as f:
            data = json.load(f)

        self.detector.burst_size = data.get('burst_size', 1)

        params = data.get('parameters', {})
        for side in SIDES:
            for band in BANDS:
                if side in params and band in params[side]:
                    self.spect_params[side][band] = SpectrogramParams.from_config_dict(params[side][band])
                    # if not in config file use burst size
                    if self.spect_params[side][band].background_burst_size is None:
                        self.spect_params[side][band].background_burst_size = self.detector.burst_size

        filters = data.get('filters', {})
        for side in SIDES:
            for band in BANDS:
                if side in filters and band in filters[side]:
                    self.filters[side][band] = FilterRange.from_config_list(filters[side][band])

        # These two were renamed: 'exclusion_filters' -> 'frequency_exclusions' and
        # 'exclusion_regions' -> 'spectrogram_masks'. Fall back to the old key names
        # so configs written before the rename still load.
        try:
            exclusions = data['frequency_exclusions']
        except KeyError:
            exclusions = data.get('exclusion_filters', {})
        for side in SIDES:
            if side in exclusions:
                self.frequency_exclusions[side] = [
                    FrequencyExclusion.from_config_list(e) for e in exclusions[side]
                ]

        try:
            masks = data['spectrogram_masks']
        except KeyError:
            masks = data.get('exclusion_regions', {})
        for side in SIDES:
            for band in BANDS:
                if side in masks and band in masks[side]:
                    self.spectrogram_masks[side][band] = [
                        SpectrogramMask.from_config_list(m) for m in masks[side][band]
                    ]


        # Older config files have no 'reconstruction_times' key; keep current values then.
        recon_times = data.get('reconstruction_times', {})
        self.reconstruction_start_time = recon_times.get('start_time', self.reconstruction_start_time)
        self.reconstruction_end_time = recon_times.get('end_time', self.reconstruction_end_time)
        self.reconstruction_time_step = recon_times.get('time_step', self.reconstruction_time_step)

        # Older config files have no 'custom_density_cutoff' key; keep current values then.
        custom_density_cutoff = data.get('custom_density_cutoff',  {})
        self.custom_density_cutoff = custom_density_cutoff.get('custom_density_cutoff', self.custom_density_cutoff)
        self.custom_density_cutoff_value = custom_density_cutoff.get('custom_density_cutoff_value', self.custom_density_cutoff_value)
        self.custom_density_cutoff_multiplier = custom_density_cutoff.get('custom_density_cutoff_multiplier', self.custom_density_cutoff_multiplier)

        print("after load:", self.frequency_exclusions)