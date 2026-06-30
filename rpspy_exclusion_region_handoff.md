# Hand-off: 2D time-gated exclusion regions in `rpspy` reconstruction

## Goal

Add support for **2D spectrogram exclusion regions** to the full profile
reconstruction in `rpspy`, mirroring a feature already implemented in the
`reflecto-lab` GUI. A region marks a rectangular area of a band's beat-frequency
spectrogram that must be **ignored when finding the maximum beat frequency**
(the peak that becomes the group delay). Each region is defined by three ranges:

- `(t_min, t_max)` — shot/discharge time in **seconds**. Gates *which sweeps* the
  region applies to.
- `(f_prob_min, f_prob_max)` — probing frequency in **Hz** (spectrogram x-axis).
- `(f_beat_min, f_beat_max)` — beat frequency in **Hz** (spectrogram y-axis).

Regions are **per band + side** (e.g. `Q-HFS`), because beat-frequency ranges are
band specific.

This is distinct from the existing `exclusion_regions` parameter, which is a 1D
list of `[low, high]` probing-frequency ranges applied *after* peak-finding (it
drops columns from the merged curve). Keep that as-is; this is a new, separate
mechanism that masks the spectrogram *before* peak-finding.

## Background: where peak-finding happens

All file references are in the `rpspy` package.

### `reconstruction.py` — `batch_processing(...)`

This computes the group delay for one band/side over the whole time window. The
relevant tail of the function:

```python
# band.burst_spectrogram has shape (n_time, n_beat, n_probe):
#   axis 0 = central sweeps (time points), axis 1 = beat freq (band.y),
#   axis 2 = probing freq (band.x)
# band.y : beat-frequency axis (Hz)         -> set in compute_spectrograms
# band.x : probing-frequency axis (Hz)      -> remapped from burst time in compute_spectrograms
# timestamps, central_sweeps : available earlier in the function
#   times = np.take(timestamps, central_sweeps)  -> shape (n_time,), seconds

# Filter spectrograms (zero the values above and below the filters)
band.burst_spectrogram[np.broadcast_to(band.y[:, None], band.burst_spectrogram.shape) <= band.dispersion + lower_filter] = band.burst_spectrogram.min()
band.burst_spectrogram[np.broadcast_to(band.y[:, None], band.burst_spectrogram.shape) >= band.dispersion + upper_filter] = band.burst_spectrogram.min()

# >>> INSERT REGION MASKING HERE <<<

# Get group delay by quadratic fitting
band.estimate_maximum_frequency, _ = quadratic_peak_interpolation(
    band.burst_spectrogram, axis=1, coordinate=band.y,
)
band.tau = band.estimate_maximum_frequency / band.sweep_rate
band.tau = band.tau - tau_offset
return np.take(timestamps, central_sweeps), band.x, band.tau
```

So `quadratic_peak_interpolation` over `axis=1` (beat freq) gives the peak per
`(time, probe)` cell. Masking must happen on `band.burst_spectrogram` between the
filter lines and this call — exactly like the existing filter masking, just for a
3D box instead of a beat-frequency band.

### `shot_reconstruction.py` — `full_profile_reconstruction(...)`

- Has the public params `filters`, `exclusion_regions` (the 1D kind), etc.
- Per-band processing runs in `_process_band`, which calls `batch_processing(...)`.
- After merge, `band.gd` is concatenated to `hfs_all_gd` / `lfs_all_gd` of shape
  `(n_time, n_freq)`; the 1D `exclusion_regions` are then applied on the freq axis.

## Changes to make in `rpspy`

### 1. `batch_processing` — accept and apply regions

Add a parameter (default empty) and apply the mask before the peak fit.

```python
def batch_processing(..., background_sweep=0, regions=None):
    ...
    if regions is None:
        regions = []
    ...
    # (after the two filter-masking lines, before quadratic_peak_interpolation)
    if regions:
        times = np.take(timestamps, central_sweeps)        # (n_time,) seconds
        floor = band.burst_spectrogram.min()
        for t_min, t_max, fp_min, fp_max, fb_min, fb_max in regions:
            t_sel  = (times  >= t_min)  & (times  <= t_max)     # axis 0
            fb_sel = (band.y >= fb_min) & (band.y <= fb_max)    # axis 1 (beat)
            fp_sel = (band.x >= fp_min) & (band.x <= fp_max)    # axis 2 (probe)
            if t_sel.any() and fb_sel.any() and fp_sel.any():
                band.burst_spectrogram[np.ix_(t_sel, fb_sel, fp_sel)] = floor
```

Notes:
- `timestamps` and `central_sweeps` are already computed earlier in
  `batch_processing` (from `get_timestamps` and `burst_mode_handler`).
- Each region is a flat 6-tuple/list `[t_min, t_max, f_prob_min, f_prob_max,
  f_beat_min, f_beat_max]`. (The `enabled` flag is filtered out caller-side, like
  `exclusion_regions` does — see `shot_reconstruction.py` exclusion list-comp.)
- `np.ix_` builds the 3D outer-product index; this masks only cells inside the box.
- Do **not** mask the full beat column (that yields NaN peaks); a region always
  leaves beat data above/below the box, so the peak fit stays valid.

### 2. `full_profile_reconstruction` — new parameter + threading

Add a new keyword param, distinct from the 1D `exclusion_regions`:

```python
def full_profile_reconstruction(..., exclusion_regions=None, ...,
                                spectrogram_regions=None, ...):
    ...
    # default: no regions for any band/side
    if spectrogram_regions is None:
        spectrogram_regions = {
            'HFS': {'K': [], 'Ka': [], 'Q': [], 'V': []},
            'LFS': {'K': [], 'Ka': [], 'Q': [], 'V': []},
        }
```

Thread it into `_process_band`:

```python
def _process_band(band):
    bg_sweep = background_sweeps[band.side][band.name]
    times, freq, gd = batch_processing(
        ..., background_sweep=bg_sweep,
        regions=spectrogram_regions[band.side][band.name],
    )
    return times, freq, gd
```

### 3. (Optional) persist regions in the output config

`write_shotfile.py` already serializes `exclusion_regions` into the saved config.
If you want the 2D regions saved alongside, add `spectrogram_regions` there too.
Not required for the masking to work.

## Verification (in `rpspy`)

1. Pick a shot/band with a known spurious peak in a time window.
2. Call `full_profile_reconstruction(..., return_profiles=True)` once with no
   regions, once with a region covering the spurious `(time, f_prob, f_beat)` box.
3. Confirm `band.tau` (group delay) changes only for the gated sweeps/probe range,
   and the resulting profile no longer reflects the spurious peak.
4. Sanity: a region whose `(t_min, t_max)` excludes all sweeps must be a no-op.
5. Sanity: no NaNs appear in `band.tau` (a region must never blank an entire beat
   column).

## Integration back in `reflecto-lab` (do this after the rpspy side is published)

The GUI already stores regions as `ExclusionRegion` per band/side
(`model/state.py`), with `to_config_list()` →
`[t_min, t_max, f_prob_min, f_prob_max, f_beat_min, f_beat_max, enabled]`.

Wire them into reconstruction:
- `model/state.py` `ReconstructionInput`: add a field, e.g.
  `exclusion_regions_2d: dict = field(default_factory=dict)`.
- `controller/app_controller.py` where `ReconstructionInput(...)` is built (next to
  `exclusion_filters=m.exclusion_filters`): pass
  `exclusion_regions_2d=m.exclusion_regions`.
- `model/reconstruction.py` `ReconstructionWorker.reconstruct`: build the dict
  rpspy expects (enabled only, drop the flag) and pass it through:

```python
spectrogram_regions = {
    side: {
        band: [r.to_config_list()[:6]
               for r in params.exclusion_regions_2d[side][band] if r.enabled]
        for band in ['K', 'Ka', 'Q', 'V']
    }
    for side in ['HFS', 'LFS']
}
rpspy.full_profile_reconstruction(..., spectrogram_regions=spectrogram_regions)
```

Pin/upgrade the `rpspy` dependency to the version that adds `spectrogram_regions`,
and keep the call backward-tolerant if you support older rpspy (e.g. only pass the
kwarg when supported).
```
