import logging
import rpspy
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from constants import DEFAULT_LINEARIZATION_SWEEP

logger = logging.getLogger(__name__)


class ReconstructionWorker(QObject):
    """Handles profile reconstruction in a separate thread.

    Receives a plain ReconstructionInput data object, not the view.
    """

    finished_signal = pyqtSignal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=None, **kwargs)

    @pyqtSlot(object)
    def reconstruct(self, params):
        """Perform full profile reconstruction.

        Args:
            params: ReconstructionInput dataclass with all needed values.
        """
        spectrogram_options = {}
        for side in ['HFS', 'LFS']:
            spectrogram_options[side] = {}
            for band in ['K', 'Ka', 'Q', 'V']:
                sp = params.spect_params[side][band]
                spectrogram_options[side][band] = {
                    'nperseg': sp.nperseg,
                    'noverlap': sp.noverlap,
                    'nfft': sp.nfft,
                }

        subtract_background_on_bands = []
        subtract_dispersion_on_bands = []
        for side in ['HFS', 'LFS']:
            for band in ['K', 'Ka', 'Q', 'V']:
                sp = params.spect_params[side][band]
                if sp.subtract_background:
                    subtract_background_on_bands.append(f"{band}-{side}")
                if sp.subtract_dispersion:
                    subtract_dispersion_on_bands.append(f"{band}-{side}")

        # Convert filters from FilterRange objects to the list format rpspy expects
        filters_dict = {}
        for side in ['HFS', 'LFS']:
            filters_dict[side] = {}
            for band in ['K', 'Ka', 'Q', 'V']:
                filt = params.filters[side][band]
                filters_dict[side][band] = [filt.low, filt.high]

        # Convert exclusion filters from ExclusionRange objects to list format
        exclusion_dict = {}
        for side in ['HFS', 'LFS']:
            exclusion_dict[side] = [[e.low, e.high] for e in params.exclusion_filters[side]]

        rpspy.full_profile_reconstruction(
            shot=params.shot,
            shotfile_dir=params.file_path,
            linearization_shotfile_dir=params.file_path,
            sweep_linearization=DEFAULT_LINEARIZATION_SWEEP,
            shot_linearization=params.shot,
            spectrogram_options=spectrogram_options,
            filters=filters_dict,
            exclusion_regions=exclusion_dict,
            subtract_background_on_bands=subtract_background_on_bands,
            subtract_dispersion_on_bands=subtract_dispersion_on_bands,
            start_time=params.start_time,
            end_time=params.end_time,
            time_step=params.time_step,
            burst=params.burst_size,
            write_private_shotfile=params.write_private_shotfile,
            write_public_shotfile=params.write_public_shotfile,
            write_hdf5=params.write_hdf5,
            hdf5_destination_path=params.destination_dir,
            return_profiles=False,
            initialization_lfs=params.get_init_lfs,
            initialization_hfs=params.get_init_hfs,
            density_cutoff=params.density_cutoff if params.apply_density_cutoff else None,
        )

        self.finished_signal.emit()
