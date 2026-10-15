from constants import BANDS, SIDES, DEFAULT_LINEARIZATION_SWEEP


def build_reconstruction_kwargs(params):
    """Arguments of ``rpspy.full_profile_reconstruction`` for a ReconstructionInput.

    Kept free of any Qt dependency, and the single place where the settings of the
    window are turned into rpspy arguments.
    """
    spectrogram_options = {
        side: {
            band: {
                'nperseg': params.spect_params[side][band].nperseg,
                'noverlap': params.spect_params[side][band].noverlap,
                'nfft': params.spect_params[side][band].nfft,
            } for band in BANDS
        } for side in SIDES
    }

    subtract_background_on_bands = []
    subtract_dispersion_on_bands = []
    for side in SIDES:
        for band in BANDS:
            sp = params.spect_params[side][band]
            if sp.subtract_background:
                subtract_background_on_bands.append(f"{band}-{side}")
            if sp.subtract_dispersion:
                subtract_dispersion_on_bands.append(f"{band}-{side}")

    background_sweeps = {
        side: {band: params.spect_params[side][band].background_sweep for band in BANDS} for side in SIDES
    }
    background_burst_size = {
        side: {band: params.spect_params[side][band].background_burst_size for band in BANDS} for side in SIDES
    }

    filters = {side: {band: params.filters[side][band].to_rpspy() for band in BANDS} for side in SIDES}

    frequency_exclusions = {
        side: [e.to_rpspy() for e in params.frequency_exclusions[side] if e.enabled] for side in SIDES
    }

    spectrogram_masks = {
        side: {
            band: [m.to_rpspy() for m in params.spectrogram_masks[side][band] if m.enabled] for band in BANDS
        } for side in SIDES
    }

    return dict(
        shot=params.shot,
        shotfile_dir=params.file_path,
        linearization_shotfile_dir=params.file_path,
        sweep_linearization=DEFAULT_LINEARIZATION_SWEEP,
        shot_linearization=params.shot,
        spectrogram_options=spectrogram_options,
        filters=filters,
        frequency_exclusions=frequency_exclusions,
        spectrogram_masks=spectrogram_masks,
        subtract_background_on_bands=subtract_background_on_bands,
        subtract_dispersion_on_bands=subtract_dispersion_on_bands,
        start_time=params.start_time,
        end_time=params.end_time,
        time_step=params.time_step,
        burst=params.burst_size,
        write_private_shotfile=params.write_private_shotfile,
        write_public_shotfile=params.write_public_shotfile,
        write_hdf5=params.write_hdf5,
        overwrite_existing_hdf5_file=True,
        hdf5_destination_path=params.hdf5_destination_path,
        return_profiles=False,
        initialization_lfs=params.get_init_lfs,
        initialization_hfs=params.get_init_hfs,
        # The cutoff is a time trace, or none. rpspy's own lookup of the interferometer signal
        # (custom_density_cutoff) is not used: what is plotted must be what is applied.
        density_cutoff_times=params.density_cutoff_times,
        density_cutoff_values=params.density_cutoff_values,
        custom_density_cutoff=None,
        background_sweeps=background_sweeps,
        background_burst_size=background_burst_size,
    )
