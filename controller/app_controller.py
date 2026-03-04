import logging
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot

from constants import (
    BANDS, SIDES, MIN_NPERSEG, MAX_NFFT,
    DECIMALS_EXCLUSIONS,
)
from model.shot_model import ShotModel
from model.state import ReconstructionInput
from model.reconstruction import ReconstructionWorker
from view.main_window import MainWindowView
from view.parameter_panels import ParameterPanels
from view.plot_renderers import PlotRenderer
from utils.helpers import round_to_nearest
from utils.ui_helpers import show_warning

logger = logging.getLogger(__name__)


class AppController(QObject):
    """Wires model and view together. Owns all signal-handling logic."""

    request_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        # Create components
        self.model = ShotModel()
        self.panels = ParameterPanels()
        self.view = MainWindowView()
        self.view.set_parameter_panels(self.panels)
        self.renderer = PlotRenderer()

        # Threading
        self._thread = QThread()
        self._worker = ReconstructionWorker()
        self._worker.finished_signal.connect(self._on_reconstruct_finished)
        self.request_signal.connect(self._worker.reconstruct)
        self._worker.moveToThread(self._thread)

        qApp = QApplication.instance()
        if qApp is not None:
            qApp.aboutToQuit.connect(self._thread.quit)
        self._thread.start()

        # Suppression flags
        self._suppress_fft_updates = False
        self._suppress_exclusions = False

        # Wire signals
        self._connect_signals()

    def _connect_signals(self):
        """All signal connections in one place."""
        p = self.panels

        # File/shot
        p.file.child('Open').sigValueChanged.connect(lambda: self._on_shot_changed('open'))
        p.file.child('Shot').sigValueChanged.connect(lambda: self._on_shot_changed('shot'))

        # Initialization
        p.init.child('Type').sigValueChanged.connect(lambda: self._on_init_changed('type'))
        p.init.child('Value').sigValueChanged.connect(lambda: self._on_init_changed('value'))
        p.init.child('File').sigValueChanged.connect(lambda: self._on_init_changed('file'))

        # Config
        p.config.child('Save').sigValueChanged.connect(self._on_save_config)
        p.config.child('Load').sigValueChanged.connect(self._on_load_config)

        # Detector
        p.detector.child('Band').sigValueChanged.connect(lambda: self._on_detector_changed('band'))
        p.detector.child('Side').sigValueChanged.connect(lambda: self._on_detector_changed('side'))

        # Sweep
        p.sweep.child('Sweep').sigValueChanged.connect(lambda: self._on_sweep_changed('slider'))
        p.sweep.child('Sweep nº').sigValueChanged.connect(lambda: self._on_sweep_changed('number'))
        p.sweep.child('Timestamp').sigValueChanged.connect(lambda: self._on_sweep_changed('timestamp'))

        # FFT params
        p.fft.child('nperseg').sigValueChanged.connect(lambda: self._on_fft_changed('nperseg'))
        p.fft.child('noverlap').sigValueChanged.connect(lambda: self._on_fft_changed('noverlap'))
        p.fft.child('nfft').sigValueChanged.connect(lambda: self._on_fft_changed('nfft'))
        p.fft.child('burst size (odd)').sigValueChanged.connect(lambda: self._on_fft_changed('burst_size'))
        p.fft.child('Scale').sigValueChanged.connect(self._on_scale_or_colormap_changed)
        p.fft.child('Subtract background').sigValueChanged.connect(lambda: self._on_fft_changed('subtract_background'))
        p.fft.child('Subtract dispersion').sigValueChanged.connect(lambda: self._on_fft_changed('subtract_dispersion'))
        p.fft.child('Color Map').sigValueChanged.connect(self._on_scale_or_colormap_changed)
        p.fft.child('Filters').child('Low Filter').sigValueChanged.connect(lambda: self._on_fft_changed('low_filter'))
        p.fft.child('Filters').child('High Filter').sigValueChanged.connect(lambda: self._on_fft_changed('high_filter'))
        p.fft.child('Exclude frequencies').sigAddNew.connect(self._on_add_exclusion)

        # Profiles
        p.profiles.child('Coordinates').sigValueChanged.connect(self._on_profile_coord_changed)

        # Reconstruct
        p.reconstruct.child('Start Time').sigValueChanged.connect(lambda: self._on_reconstruct_params_changed('start'))
        p.reconstruct.child('End Time').sigValueChanged.connect(lambda: self._on_reconstruct_params_changed('end'))
        p.reconstruct.child('Reconstruct Shot').sigActivated.connect(self._on_request_reconstruct)
        p.reconstruct.child('Apply Custom Density Cutoff').sigValueChanged.connect(self._on_cutoff_changed)

    # --- Helpers to sync model ↔ panels ---

    def _sync_detector_from_panels(self):
        """Read current detector selection from panels into model."""
        self.model.detector.band = self.panels.detector.child('Band').value()
        self.model.detector.side = self.panels.detector.child('Side').value()

    def _sync_params_to_panels(self):
        """Update panel values to reflect current band/side's stored params."""
        m = self.model
        p = self.panels
        d = m.detector
        sp = m.spect_params[d.side][d.band]
        filt = m.filters[d.side][d.band]

        p.fft.child('Filters').child('Low Filter').setValue(filt.low, blockSignal=self._on_fft_changed_blocked)
        p.fft.child('Filters').child('High Filter').setValue(filt.high, blockSignal=self._on_fft_changed_blocked)
        p.fft.child('nperseg').setValue(sp.nperseg, blockSignal=self._on_fft_changed_blocked)

        # Update noverlap limits then value
        p.fft.child('noverlap').sigValueChanged.disconnect()
        p.fft.child('noverlap').setLimits((0, sp.nperseg - 1))
        p.fft.child('noverlap').sigValueChanged.connect(lambda: self._on_fft_changed('noverlap'))
        p.fft.child('noverlap').setValue(sp.noverlap, blockSignal=self._on_fft_changed_blocked)

        # Update nfft limits then value
        p.fft.child('nfft').sigValueChanged.disconnect()
        p.fft.child('nfft').setLimits((sp.nperseg, np.inf))
        p.fft.child('nfft').sigValueChanged.connect(lambda: self._on_fft_changed('nfft'))
        p.fft.child('nfft').setValue(sp.nfft, blockSignal=self._on_fft_changed_blocked)

        p.fft.child('Subtract background').setValue(sp.subtract_background, blockSignal=self._on_fft_changed_blocked)
        p.fft.child('Subtract dispersion').setValue(sp.subtract_dispersion if sp.subtract_dispersion is not None else False, blockSignal=self._on_fft_changed_blocked)

        # Update exclusion filter UI
        p.fft.child('Exclude frequencies').clearChildren()
        self._suppress_exclusions = True
        for excl in m.exclusion_filters[d.side]:
            self._on_add_exclusion()
            children = p.fft.child('Exclude frequencies').children()
            children[-1].child('from').setValue(excl.low, blockSignal=self._on_exclusion_changed_blocked)
            children[-1].child('to').setValue(excl.high, blockSignal=self._on_exclusion_changed_blocked)
        self._suppress_exclusions = False

        # Update initialization panels
        iv = m.init_values[d.side]
        p.init.child('Type').setValue(iv.type, blockSignal=self._on_init_changed_blocked)
        p.init.child('Value').setValue(iv.current_value, blockSignal=self._on_init_changed_blocked)
        p.init.child('File').setValue(iv.file.name, blockSignal=self._on_init_changed_blocked)

        if iv.type == 'Default (recommended)':
            p.init.child('Value').setOpts(readonly=True)
            p.init.child('File').setOpts(visible=False)
        elif iv.type == 'Custom':
            p.init.child('Value').setOpts(readonly=False)
            p.init.child('File').setOpts(visible=False)
        elif iv.type == 'From file':
            p.init.child('Value').setOpts(readonly=True)
            p.init.child('File').setOpts(visible=True)

    # Dummy callables for blockSignal (lambdas can't be used as blockSignal targets)
    def _on_fft_changed_blocked(self):
        pass

    def _on_exclusion_changed_blocked(self):
        pass

    def _on_init_changed_blocked(self):
        pass

    # --- Shot loading ---

    def _on_shot_changed(self, source):
        """Handles shot/file load."""
        p = self.panels
        m = self.model

        if source == 'open':
            file_path = p.file.child('Open').value()
            try:
                m.load_shot_from_path(file_path)
                p.file.child('Shot').setValue(m.shot, blockSignal=lambda: self._on_shot_changed('shot'))
            except (ValueError, FileNotFoundError):
                show_warning()
                return
            finally:
                p.file.child('Open').setValue(m.file_path, blockSignal=lambda: self._on_shot_changed('open'))

        elif source == 'shot':
            shot = p.file.child('Shot').value()
            try:
                m.load_shot_from_number(shot)
                p.file.child('Open').setValue(m.file_path, blockSignal=lambda: self._on_shot_changed('open'))
            except FileNotFoundError:
                show_warning()
                p.file.child('Shot').setValue(m.shot, blockSignal=lambda: self._on_shot_changed('shot'))
                return

        # Show all parameters
        self.view.show_post_load_params(self.panels)

        # Except parameters that default to hidden states
        p.reconstruct.child('Density Cutoff').setOpts(readonly=True)
        p.reconstruct.child('Density Cutoff').setOpts(visible=False)

        # Sync detector
        self._sync_detector_from_panels()
        d = m.detector

        # Post-load init
        m.post_load_init()
        m.initialize_limiters(d.side, p.sweep.child('Timestamp').value(), define_init_funcs=True)

        # Full recompute + draw
        self._recompute_and_draw_all()

        # Set parameter limits
        ts = m.time_stamps
        p.sweep.child('Sweep').setLimits((1, len(ts)))
        p.sweep.child('Sweep nº').setLimits((1, len(ts)))
        p.sweep.child('Timestamp').setLimits((ts[0], ts[-1]))
        p.sweep.child('Timestamp').setOpts(step=ts[1])
        p.fft.child('nperseg').setLimits((MIN_NPERSEG, len(m.current_sweep.data) // 2))
        p.fft.child('noverlap').setLimits((0, p.fft.child('nperseg').value() - 1))
        p.fft.child('nfft').setLimits((p.fft.child('nperseg').value(), MAX_NFFT))
        p.fft.child('Filters').child('Low Filter').setLimits((0, np.inf))
        fft = m.current_fft
        p.fft.child('Filters').child('High Filter').setLimits((abs(fft.f_beat[0] - fft.f_beat[1]), np.inf))
        p.reconstruct.child('Start Time').setLimits((0, len(ts) * (ts[1] - ts[0])))
        p.reconstruct.child('End Time').setLimits((0, len(ts) * (ts[1] - ts[0])))

    # --- Initialization ---

    def _on_init_changed(self, source):
        """Handle initialization parameter changes."""
        p = self.panels
        m = self.model
        d = m.detector

        if source == 'type':
            value = p.init.child('Type').value()
            m.init_values[d.side].type = value

            if value == 'Default (recommended)':
                p.init.child('Value').setOpts(readonly=True)
                p.init.child('File').setOpts(visible=False)
            elif value == 'Custom':
                p.init.child('Value').setOpts(readonly=False)
                p.init.child('File').setOpts(visible=False)
            elif value == 'From file':
                p.init.child('Value').setOpts(readonly=True)
                p.init.child('File').setOpts(visible=True)

        elif source == 'value':
            m.init_values[d.side].custom_value = p.init.child('Value').value()

        elif source == 'file':
            path = p.init.child('File').value()
            data = np.atleast_2d(np.loadtxt(path, delimiter=','))
            m.init_values[d.side].file.time = data[:, 0]
            m.init_values[d.side].file.position = data[:, 1]
            p.init.child('File').setValue(path + ' (loaded)', blockSignal=self._on_init_changed_blocked)
            m.init_values[d.side].file.name = p.init.child('File').value()

        m.initialize_limiters(d.side, p.sweep.child('Timestamp').value(), define_init_funcs=True)
        self._draw_group_delays()
        self._draw_profile()

    # --- Detector ---

    def _on_detector_changed(self, source):
        """Band/side change."""
        p = self.panels
        self._sync_detector_from_panels()
        self._sync_params_to_panels()

        if source == 'band':
            if p.detector.child('Band').value() == 'V':
                p.fft.child('Subtract dispersion').setOpts(enabled=True)
            else:
                p.fft.child('Subtract dispersion').setOpts(enabled=False)

        self._recompute_sweep()
        self._recompute_fft_and_display()

    # --- Sweep navigation ---

    def _on_sweep_changed(self, source):
        """Sweep/timestamp change."""
        p = self.panels
        m = self.model
        ts = m.time_stamps

        if source == 'slider':
            value = p.sweep.child('Sweep').value()
            m.detector.sweep = value - 1
            timestamp = ts[value - 1]
            p.sweep.child('Sweep nº').setValue(value, blockSignal=lambda: self._on_sweep_changed('number'))
            p.sweep.child('Timestamp').setValue(timestamp, blockSignal=lambda: self._on_sweep_changed('timestamp'))

        elif source == 'number':
            value = int(p.sweep.child('Sweep nº').value())
            m.detector.sweep = value - 1
            timestamp = ts[value - 1]
            p.sweep.child('Sweep nº').setValue(value, blockSignal=lambda: self._on_sweep_changed('number'))
            p.sweep.child('Sweep').setValue(value, blockSignal=lambda: self._on_sweep_changed('slider'))
            p.sweep.child('Timestamp').setValue(timestamp, blockSignal=lambda: self._on_sweep_changed('timestamp'))

        elif source == 'timestamp':
            value = p.sweep.child('Timestamp').value()
            timestamp = round_to_nearest(value, ts)
            index = np.where(ts == timestamp)
            m.detector.sweep = index[0][0]
            p.sweep.child('Timestamp').setValue(timestamp, blockSignal=lambda: self._on_sweep_changed('timestamp'))
            p.sweep.child('Sweep').setValue(index[0][0] + 1, blockSignal=lambda: self._on_sweep_changed('slider'))
            p.sweep.child('Sweep nº').setValue(index[0][0] + 1, blockSignal=lambda: self._on_sweep_changed('number'))

        m.initialize_limiters(m.detector.side, p.sweep.child('Timestamp').value())

        self._recompute_sweep()

        if not self._suppress_fft_updates:
            self._recompute_fft_and_display()
            m.compute_all_beatf()
            self._draw_group_delays()
            self._draw_profile()

    # --- FFT parameters ---

    def _on_fft_changed(self, source):
        """FFT param change."""
        p = self.panels
        m = self.model
        d = m.detector
        sp = m.spect_params[d.side][d.band]
        filt = m.filters[d.side][d.band]
        fft = m.current_fft

        if source == 'nperseg':
            value = int(p.fft.child('nperseg').value())
            sp.nperseg = value
            p.fft.child('nperseg').setValue(value, blockSignal=self._on_fft_changed_blocked)
            self._suppress_fft_updates = True
            p.fft.child('noverlap').setLimits((0, value - 1))
            p.fft.child('nfft').setLimits((value, np.inf))
            self._suppress_fft_updates = False

        elif source == 'noverlap':
            value = int(p.fft.child('noverlap').value())
            sp.noverlap = value
            p.fft.child('noverlap').setValue(value, blockSignal=self._on_fft_changed_blocked)

        elif source == 'nfft':
            value = int(p.fft.child('nfft').value())
            sp.nfft = value
            p.fft.child('nfft').setValue(value, blockSignal=self._on_fft_changed_blocked)

        elif source == 'burst_size':
            value = int(p.fft.child('burst size (odd)').value())
            if value % 2 == 0:
                value -= 1
            p.fft.child('burst size (odd)').setValue(value, blockSignal=self._on_fft_changed_blocked)
            m.detector.burst_size = value

            lower_limit = 1 + value // 2
            upper_limit = len(m.time_stamps) - value // 2
            self._suppress_fft_updates = True
            p.sweep.child('Sweep').setLimits((lower_limit, upper_limit))
            p.sweep.child('Sweep nº').setLimits((lower_limit, upper_limit))
            p.sweep.child('Timestamp').setLimits((m.time_stamps[lower_limit - 1], m.time_stamps[upper_limit - 1]))
            self._suppress_fft_updates = False

        elif source == 'subtract_background':
            sp.subtract_background = p.fft.child('Subtract background').value()

        elif source == 'subtract_dispersion':
            sp.subtract_dispersion = p.fft.child('Subtract dispersion').value()

        elif source == 'low_filter':
            filt.low = p.fft.child('Filters').child('Low Filter').value()
            if fft.f_beat is not None and filt.low + abs(fft.f_beat[1] - fft.f_beat[0]) >= filt.high:
                filt.high = filt.low + abs(fft.f_beat[1] - fft.f_beat[0])
                p.fft.child('Filters').child('High Filter').setValue(filt.high, blockSignal=self._on_fft_changed_blocked)

        elif source == 'high_filter':
            filt.high = p.fft.child('Filters').child('High Filter').value()
            if fft.f_beat is not None and filt.high - abs(fft.f_beat[1] - fft.f_beat[0]) <= filt.low:
                filt.low = filt.high - abs(fft.f_beat[1] - fft.f_beat[0])
                p.fft.child('Filters').child('Low Filter').setValue(filt.low, blockSignal=self._on_fft_changed_blocked)

        # Trigger appropriate updates
        if not self._suppress_fft_updates:
            if source in ('low_filter', 'high_filter', 'subtract_background'):
                self._draw_spectrogram()
                m.compute_one_beatf(d.band, d.side)
            elif source == 'burst_size':
                for side in SIDES:
                    for band in BANDS:
                        m.compute_background(band, side)
                self._recompute_fft_and_display()
                m.compute_all_beatf()
            else:
                m.compute_background(d.band, d.side)
                self._recompute_fft_and_display()
                m.compute_one_beatf(d.band, d.side)

            self._draw_group_delays()
            self._draw_profile()

    # --- Exclusions ---

    def _on_add_exclusion(self):
        """Add a new exclusion frequency range."""
        from model.state import ExclusionRange
        p = self.panels
        m = self.model
        d = m.detector

        pos = len(p.fft.child('Exclude frequencies').children())
        if pos < 10:
            p.fft.child('Exclude frequencies').addChild({
                'name': f'{pos + 1}', 'type': 'group', 'children': [
                    {'name': 'from', 'type': 'float', 'value': 0, 'suffix': 'Hz', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'to', 'type': 'float', 'value': 0, 'suffix': 'Hz', 'siPrefix': True, 'decimals': DECIMALS_EXCLUSIONS},
                    {'name': 'Remove', 'type': 'action'},
                ]
            })

            child = p.fft.child('Exclude frequencies').child(f'{pos + 1}')
            child.child('from').sigValueChanged.connect(self._on_exclusion_changed)
            child.child('to').sigValueChanged.connect(self._on_exclusion_changed)
            child.child('Remove').sigActivated.connect(self._on_remove_exclusion)

            if not self._suppress_exclusions:
                m.exclusion_filters[d.side].append(ExclusionRange(0.0, 0.0))

    def _on_remove_exclusion(self):
        """Remove an exclusion frequency range."""
        sender = self.sender()

        parent = sender.parent()
        num_of_parent = int(parent.name())
        p = self.panels
        m = self.model
        d = m.detector

        p.fft.child('Exclude frequencies').removeChild(parent)

        # Renumber remaining exclusions
        for i in range(num_of_parent, len(p.fft.child('Exclude frequencies').children()) + 1):
            p.fft.child('Exclude frequencies').child(f'{i + 1}').setName(f'{i}')

        # Remove from model
        m.exclusion_filters[d.side].pop(num_of_parent - 1)

        self._draw_spectrogram_beatf_overlay()
        self._draw_group_delays()
        self._draw_profile()

    def _on_exclusion_changed(self):
        """Exclusion range value changed."""
        sender = self.sender()

        if sender.name() == 'from':
            if sender.value() > sender.parent().child('to').value():
                sender.parent().child('to').setValue(sender.value(), blockSignal=self._on_exclusion_changed_blocked)
        elif sender.name() == 'to':
            if sender.value() < sender.parent().child('from').value():
                sender.parent().child('from').setValue(sender.value(), blockSignal=self._on_exclusion_changed_blocked)

        exclusion_num = int(sender.parent().name())
        d = self.model.detector
        self.model.exclusion_filters[d.side][exclusion_num - 1].low = sender.parent().child('from').value()
        self.model.exclusion_filters[d.side][exclusion_num - 1].high = sender.parent().child('to').value()

        self._draw_spectrogram_beatf_overlay()
        self._draw_group_delays()
        self._draw_profile()

    # --- Visual-only changes ---

    def _on_scale_or_colormap_changed(self):
        """Scale or colormap change — just redraw, no recompute."""
        self._draw_spectrogram()

    def _on_profile_coord_changed(self):
        """Profile coordinate mode change."""
        self._draw_profile()

    # --- Reconstruct params ---

    def _on_reconstruct_params_changed(self, source):
        """Validate start/end times."""
        p = self.panels
        if source == 'start':
            if p.reconstruct.child('Start Time').value() > p.reconstruct.child('End Time').value():
                p.reconstruct.child('End Time').setValue(
                    p.reconstruct.child('Start Time').value(),
                    blockSignal=lambda: self._on_reconstruct_params_changed('end'),
                )
        elif source == 'end':
            if p.reconstruct.child('End Time').value() < p.reconstruct.child('Start Time').value():
                p.reconstruct.child('Start Time').setValue(
                    p.reconstruct.child('End Time').value(),
                    blockSignal=lambda: self._on_reconstruct_params_changed('start'),
                )

    def _on_cutoff_changed(self):
        """Toggle density cutoff visibility."""
        enabled = self.panels.reconstruct.child('Apply Custom Density Cutoff').value()
        self.panels.reconstruct.child('Density Cutoff').setOpts(readonly=not enabled)
        self.panels.reconstruct.child('Density Cutoff').setOpts(visible=enabled)

    # --- Reconstruction ---

    def _on_request_reconstruct(self):
        """Request reconstruction in separate thread."""
        m = self.model
        p = self.panels

        params = ReconstructionInput(
            shot=m.shot,
            file_path=m.file_path,
            spect_params=m.spect_params,
            filters=m.filters,
            exclusion_filters=m.exclusion_filters,
            burst_size=m.detector.burst_size,
            start_time=p.reconstruct.child('Start Time').value(),
            end_time=p.reconstruct.child('End Time').value(),
            time_step=p.reconstruct.child('Time Step').value(),
            apply_density_cutoff=p.reconstruct.child('Apply Custom Density Cutoff').value(),
            density_cutoff=p.reconstruct.child('Density Cutoff').value(),
            write_private_shotfile=p.reconstruct.child('Reconstruction Output').child('Private Shotfile').value(),
            write_public_shotfile=p.reconstruct.child('Reconstruction Output').child('Public Shotfile').value(),
            write_hdf5=p.reconstruct.child('Reconstruction Output').child('HDF5').value(),
            get_init_lfs=lambda time: m.get_init('LFS', time),
            get_init_hfs=lambda time: m.get_init('HFS', time),
        )

        self.request_signal.emit(params)
        self.view.set_reconstruct_ui_enabled(self.panels, False)

    @pyqtSlot()
    def _on_reconstruct_finished(self):
        """Re-enable UI after reconstruction."""
        self.view.set_reconstruct_ui_enabled(self.panels, True)

    # --- Config ---

    def _on_save_config(self):
        """Save current configuration."""
        path = self.panels.config.child('Save').value()
        self.model.save_config(path)
        self.panels.config.child('Save').setValue(path + ' (saved)', blockSignal=self._on_save_config)
        self.panels.config.child('Load').setValue('', blockSignal=self._on_load_config)

    def _on_load_config(self):
        """Load configuration from file."""
        path = self.panels.config.child('Load').value()
        self.model.load_config(path)
        self._sync_params_to_panels()

        # Force change signal to handle sweep number and plot everything
        p = self.panels
        p.fft.child('burst size (odd)').setValue(2, blockSignal=self._on_fft_changed_blocked)
        p.fft.child('burst size (odd)').setValue(self.model.detector.burst_size)

        p.config.child('Load').setValue(path + ' (loaded)', blockSignal=self._on_load_config)
        p.config.child('Save').setValue('', blockSignal=self._on_save_config)

    # --- Recompute + redraw orchestration ---

    def _recompute_and_draw_all(self):
        """Full cascade after shot load."""
        self._recompute_sweep()
        self._recompute_fft_and_display()
        self.model.compute_all_beatf()
        self._draw_group_delays()
        self._draw_profile()

    def _recompute_sweep(self):
        """Compute sweep data and render."""
        self.model.compute_sweep()
        sw = self.model.current_sweep
        self.renderer.draw_sweep(self.view.plot_sweep, sw.x_data, sw.data, sw.signal_type)

    def _recompute_fft_and_display(self):
        """Compute FFT + display data and render spectrogram with all overlays."""
        self.model.compute_current_fft()
        self.model.compute_current_display()
        self._draw_spectrogram()

    def _draw_spectrogram(self):
        """Full spectrogram redraw including all overlays."""
        m = self.model
        d = m.detector
        fft = m.current_fft
        disp = m.current_display
        sp = m.spect_params[d.side][d.band]
        filt = m.filters[d.side][d.band]

        # Recompute display data (background subtraction + beatf) for visual-only changes
        m.compute_current_display()

        scale = self.panels.fft.child('Scale').value()
        colormap = self.panels.fft.child('Color Map').value()

        self.view.colorBar = self.renderer.draw_spectrogram(
            self.view.plot_spect, fft.Sxx, scale, colormap,
            sp.nperseg, sp.noverlap,
            fft.f, fft.f_beat, fft.f_probe, fft.fs, d.band,
            self.view.colorBar,
        )

        self.renderer.draw_dispersion_line(self.view.plot_spect, fft.f_probe, disp.y_dis)
        self.renderer.draw_filter_lines(self.view.plot_spect, fft.f_probe, disp.y_dis, filt.low, filt.high)
        self.renderer.draw_beatf_on_spectrogram(
            self.view.plot_spect, fft.f_probe, disp.y_beatf,
            m.exclusion_filters[d.side], d.side,
        )

    def _draw_spectrogram_beatf_overlay(self):
        """Redraw just the beat frequency overlay on the spectrogram (after exclusion changes)."""
        # For exclusion changes we need to recompute the beat frequency since filters may have changed
        m = self.model
        d = m.detector
        fft = m.current_fft
        disp = m.current_display

        # We need a full spectrogram redraw since pyqtgraph doesn't support selective overlay removal
        self._draw_spectrogram()

    def _draw_group_delays(self):
        """Compute aggregated delays and render group delay plot."""
        m = self.model
        m.compute_aggregated_delays()
        self.renderer.draw_group_delays(
            self.view.plot_beatf, m.beat_frequencies, m.exclusion_filters,
            m.aggregated_hfs, m.aggregated_lfs,
        )

    def _draw_profile(self):
        """Compute and render density profile."""
        m = self.model
        p = self.panels
        coord = p.profiles.child('Coordinates').value()
        timestamp = p.sweep.child('Timestamp').value()

        r_HFS, ne_HFS, r_LFS, ne_LFS = m.compute_profile(coord, timestamp)
        self.renderer.draw_profile(self.view.plot_profile, r_HFS, ne_HFS, r_LFS, ne_LFS, coord)
