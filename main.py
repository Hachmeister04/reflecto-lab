import sys
import numpy as np
import json
from scipy.signal import spectrogram
from scipy.fft import fftshift
from PyQt5.QtWidgets import QApplication, QMainWindow, QHeaderView
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.parametertree import Parameter, ParameterTree
import rpspy
import func_aux
import time


#TODO: Set limits of range view for the other graphs
#TODO: Remove hardcoded values and add them here
#TODO: Comment EVERYTHINGS

# Development option
SANDBOX = True  # If True, the program will be able to use sandbox data from local shots

# Window
WINDOW_SIZE = (1600, 800)
PARAMETER_TREE_WIDTH_PROPORTION = 0.3
GRAPH_WIDTH_PROPORTION = 0.35

# Parameter Tree
DEFAULT_SECTION_SIZE = 200

# Spectogram Params
MAX_BURST_SIZE = 285
DEFAULT_NPERSEG = 256
DEFAULT_NOVERLAP = 220
DEFAULT_NFFT = 512
MIN_NPERSEG = 10
MAX_NFFT = np.inf
DEFAULT_BURST_SIZE = 1

# Filter Params
DEFAULT_FILTER_LOW = 0 #Hz
DEFAULT_FILTER_HIGH = 10*1e6 #Hz

# Reconstruct Params
DEFAULT_START_TIME = 0 #s
DEFAULT_END_TIME = 10 #s
DEFAULT_TIMESTEP = 1e-3 #s

# Plot Params
DECIMALS_SWEEP_NUM = 6
DECIMALS_TIMESTAMP = 8

# Profile Properties
PROFILE_INVERSION_RESOLUTION = 150 #points

#TODO: Segment the code
class PlotWindow(QMainWindow):
    request_signal = pyqtSignal()
    
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        #Initiate separate thread
        self._thread = QThread()
        self._threaded = Threaded()
        self._threaded.finished_signal.connect(self.finished_reconstruct)
        self.request_signal.connect(self._threaded.reconstruct)
        self._threaded.moveToThread(self._thread)

        qApp = QApplication.instance()
        if qApp is not None:
            qApp.aboutToQuit.connect(self._thread.quit)
        self._thread.start()

        # Define some useful attributes
        self.params_added = False
        self.supress_updates_ffts = False #Prevents repeated call of the update functions from the update_fft_params function
        
        #Store the spectrogram parameters
        self.spect_params = {
            'HFS': {
                'K': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract background': False,
                    'subtract dispersion': None
                },
                'Ka': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract background': False,
                    'subtract dispersion': None
                },
                'Q': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract background': False,
                    'subtract dispersion': None
                },
                'V': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract background': False,
                    'subtract dispersion': True
                }
            },
            'LFS': {
                'K': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract background': False,
                    'subtract dispersion': None
                },
                'Ka': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract background': False,
                    'subtract dispersion': None
                },
                'Q': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract background': False,
                    'subtract dispersion': None
                },
                'V': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract background': False,
                    'subtract dispersion': True
                }
            }
        }

        # Store the burst size (global)
        self.burst_size = DEFAULT_BURST_SIZE

        #Store the beat frquencies of the 8 detectors
        self.beat_frequencies = {
            'HFS': {
                'K': [None, None, None], # [[f_probe], [beatf], [beat_time]]
                'Ka': [None, None, None],
                'Q': [None, None, None],
                'V': [None, None, None]
            },
            'LFS': {
                'K': [None, None, None],
                'Ka': [None, None, None],
                'Q': [None, None, None],
                'V': [None, None, None]
                }
            }

        #Store the filters
        self.filters = {
            'HFS': {
                'K': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'Ka': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'Q': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'V': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH]
            },
            'LFS': {
                'K': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'Ka': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'Q': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH],
                'V': [DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH]
            }
        }

        #Store de background spectrograms
        self.background_spectrograms = {
            'HFS': {
                'K': None,
                'Ka': None,
                'Q': None,
                'V': None
            },
            'LFS': {
                'K': None,
                'Ka': None,
                'Q': None,
                'V': None
            }
        }

        #---------------------------------------------------------------------------

        # Set the application icon
        self.setWindowIcon(QtGui.QIcon('reflecto-lab.png'))

        # Set up the main window
        self.setWindowTitle('ReflectoLab')
        self.setGeometry(100, 100, WINDOW_SIZE[0], WINDOW_SIZE[1])
        
        # Create docks -------------------------------------------------------------

        self.area = DockArea()

        # Create docks
        self.dock_tree = Dock("Settings", size=(WINDOW_SIZE[0]*PARAMETER_TREE_WIDTH_PROPORTION, WINDOW_SIZE[1]))
        self.dock_sweep = Dock(" ", size=(WINDOW_SIZE[0]*GRAPH_WIDTH_PROPORTION, WINDOW_SIZE[1]/2))
        self.dock_spect = Dock(" ", size=(WINDOW_SIZE[0]*GRAPH_WIDTH_PROPORTION, WINDOW_SIZE[1]/2))
        self.dock_beatf = Dock(" ", size=(WINDOW_SIZE[0]*GRAPH_WIDTH_PROPORTION, WINDOW_SIZE[1]/2))
        self.dock_profile = Dock(" ", size=(WINDOW_SIZE[0]*GRAPH_WIDTH_PROPORTION, WINDOW_SIZE[1]/2))

        # Add docks to the area
        self.area.addDock(self.dock_tree, 'left')
        self.area.addDock(self.dock_sweep, 'right', self.dock_tree)
        self.area.addDock(self.dock_spect, 'bottom', self.dock_sweep)
        self.area.addDock(self.dock_profile, 'right')
        self.area.addDock(self.dock_beatf, 'bottom', self.dock_profile)

        # Set a central widget
        self.setCentralWidget(self.area)

        # Create all the plots and add to docks ------------------------------------

        self.plot_sweep = pg.PlotWidget(title="Sweep")
        self.dock_sweep.addWidget(self.plot_sweep)

        self.plot_spect = pg.PlotWidget(title="Spectrogram")
        self.dock_spect.addWidget(self.plot_spect)

        self.plot_beatf = pg.PlotWidget(title="Beat Frequencies")
        self.dock_beatf.addWidget(self.plot_beatf)

        self.plot_profile = pg.PlotWidget(title="Profile")
        self.dock_profile.addWidget(self.plot_profile)
        
        # Create widgets -----------------------------------------------------------

        # Create the parameter tree
        self.param_tree = ParameterTree()
        self.dock_tree.addWidget(self.param_tree)

        # Set adjustable scale for names and values of the parameter tree
        self.param_tree.header().setDefaultSectionSize(DEFAULT_SECTION_SIZE)
        self.param_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        # Create parameters
        self.params_file = Parameter.create(name='File', type='group', children=[
            {'name': 'Open', 'type': 'file', 'value': None, 'fileMode': 'Directory'},
            {'name': 'Shot', 'type': 'int'},
        ])
        self.params_file.child('Open').setValue('')

        self.params_config = Parameter.create(name='Configuration', type='group', visible=False, children=[
            {'name': 'Save', 'type': 'file', 'value': None, 'fileMode': 'AnyFile', 'acceptMode': 'AcceptSave', 'nameFilter': 'JSON Files (*.json)'},
            {'name': 'Load', 'type': 'file', 'value': None, 'fileMode': 'AnyFile', 'acceptMode': 'AcceptOpen', 'nameFilter': 'JSON Files (*.json)'}
        ])
        self.params_config.child('Save').setValue('')
        self.params_config.child('Load').setValue('')

        self.params_detector = Parameter.create(name='Detector', type='group', visible=False, children=[
            {'name': 'Band', 'type': 'list', 'limits': ['K', 'Ka', 'Q', 'V']},
            {'name': 'Side', 'type': 'list', 'limits': ['HFS', 'LFS']}
        ])
        self.params_sweep = Parameter.create(name='Sweep', type='group', visible=False, children=[
            {'name': 'Sweep nº', 'type': 'float', 'value': 1, 'decimals': DECIMALS_SWEEP_NUM, 'delay': 0},
            {'name': 'Sweep', 'title': ' ', 'type': 'slider', 'limits': (1, 1)},
            {'name': 'Timestamp', 'type': 'float', 'value': 0, 'suffix': 's', 'decimals': DECIMALS_TIMESTAMP, 'siPrefix': True, 'delay': 0},
        ])
        self.params_fft = Parameter.create(name='Spectrogram', type='group', visible=False, children=[
            {'name': 'nperseg', 'type': 'float', 'value': DEFAULT_NPERSEG, 'delay': 0},
            {'name': 'noverlap', 'type': 'float', 'value': DEFAULT_NOVERLAP, 'delay': 0},
            {'name': 'nfft', 'type': 'float', 'value': DEFAULT_NFFT, 'delay': 0},
            {'name': 'burst size (odd)', 'type': 'float', 'value': DEFAULT_BURST_SIZE, 'limits': (1, MAX_BURST_SIZE), 'step': 2, 'delay': 0},
            {'name': 'Scale', 'type': 'checklist', 'limits': ['Normalized', 'Linear', 'Logarithmic'], 'exclusive': True, 'delay': 0},
            {'name': 'Subtract background', 'type': 'bool', 'value': False, 'delay': 0},
            {'name': 'Subtract dispersion', 'type': 'bool', 'value': False, 'enabled': False, 'delay': 0},
            {'name': 'Color Map', 'type': 'cmaplut', 'value': 'plasma'}
        ])
        self.params_filter = Parameter.create(name='Filters (above dispersion)', type='group', visible=False, children=[
            {'name': 'Low Filter', 'type': 'float', 'value': DEFAULT_FILTER_LOW, 'suffix': 'Hz', 'siPrefix': True, 'delay': 0},
            {'name': 'High Filter', 'type': 'float', 'value': DEFAULT_FILTER_HIGH, 'suffix': 'Hz', 'siPrefix': True, 'delay': 0}
        ])
        self.params_reconstruct = Parameter.create(name='Reconstruct Shot', type='group', visible=False, children=[
            {'name': 'Start Time', 'type': 'float', 'value': DEFAULT_START_TIME, 'suffix': 's', 'siPrefix': True, 'delay': 0},
            {'name': 'End Time', 'type': 'float', 'value': DEFAULT_END_TIME, 'suffix': 's', 'siPrefix': True, 'delay': 0},
            {'name': 'Time Step', 'type': 'float', 'value': DEFAULT_TIMESTEP, 'suffix': 's', 'siPrefix': True, 'delay': 0},
            {'name': 'Reconstruct Shot', 'type': 'action'}
        ])

        self.param_tree.addParameters(self.params_file)
        self.param_tree.addParameters(self.params_config)
        self.param_tree.addParameters(self.params_detector)
        self.param_tree.addParameters(self.params_sweep)
        self.param_tree.addParameters(self.params_fft)
        self.param_tree.addParameters(self.params_filter)
        self.param_tree.addParameters(self.params_reconstruct)

        # Connect the parameters to the functions-----------------------------------

        # Connect the file to the function that displays the rest of the parameters and the graphs
        self.params_file.child('Open').sigValueChanged.connect(self.update_shot)
        self.params_file.child('Shot').sigValueChanged.connect(self.update_shot)

        # Connect the save and load buttons to the handling functions
        self.params_config.child('Save').sigValueChanged.connect(self.save_config)
        self.params_config.child('Load').sigValueChanged.connect(self.load_config)

        # Connect the lists to update the plot
        self.params_detector.child('Band').sigValueChanged.connect(self.update_detector_params)
        self.params_detector.child('Side').sigValueChanged.connect(self.update_detector_params)

        # Connect the slider, sweep, and timestamp to update the plot
        self.params_sweep.child('Sweep').sigValueChanged.connect(self.update_plot_params)
        self.params_sweep.child('Sweep nº').sigValueChanged.connect(self.update_plot_params)
        self.params_sweep.child('Timestamp').sigValueChanged.connect(self.update_plot_params)

        #Connect the fft params to update the fft
        self.params_fft.child('nperseg').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('noverlap').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('nfft').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('burst size (odd)').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('Scale').sigValueChanged.connect(self.draw_spectrogram)
        self.params_fft.child('Subtract background').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('Subtract dispersion').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('Color Map').sigValueChanged.connect(self.draw_spectrogram)

        #Connect the filter params to update the fft
        self.params_filter.child('Low Filter').sigValueChanged.connect(self.update_fft_params)
        self.params_filter.child('High Filter').sigValueChanged.connect(self.update_fft_params)

        #Connect the start and end times to eachother
        self.params_reconstruct.child('Start Time').sigValueChanged.connect(self.update_reconstruct_params)
        self.params_reconstruct.child('End Time').sigValueChanged.connect(self.update_reconstruct_params)

        #Connect the button to reconstruct shot
        self.params_reconstruct.child('Reconstruct Shot').sigActivated.connect(self.request_reconstruct)


    def update_shot(self):
        sender = self.sender()

        # Name the path to the directory
        if sender == self.params_file.child('Open'):
            self.file_path = self.params_file.child('Open').value()
            self.shot = func_aux.get_shot_from_path(self.file_path)
            self.params_file.child('Shot').setValue(self.shot, blockSignal=self.update_shot)
        elif sender == self.params_file.child('Shot'):
            self.file_path = func_aux.get_path_from_shot(self.params_file.child('Shot').value())
            self.shot = self.params_file.child('Shot').value()
            self.params_file.child('Open').setValue(self.file_path, blockSignal=self.update_shot)

        # Show all parameters-----------------------------------------------------------

        self.params_config.setOpts(visible=True)
        self.params_detector.setOpts(visible=True)
        self.params_sweep.setOpts(visible=True)
        self.params_fft.setOpts(visible=True)
        self.params_filter.setOpts(visible=True)
        self.params_reconstruct.setOpts(visible=True)

        # Set selfs before the plots-----------------------------------------------------

        self.band = self.params_detector.child('Band').value()
        self.side = self.params_detector.child('Side').value()
        self.sweep = int(self.params_sweep.child('Sweep nº').value()) - 1
        self.nperseg = int(self.params_fft.child('nperseg').value())
        self.noverlap = int(self.params_fft.child('noverlap').value())
        self.nfft = int(self.params_fft.child('nfft').value())
        self.burst_size = int(self.params_fft.child('burst size (odd)').value())
        self.subtract_background = self.params_fft.child('Subtract background').value()
        self.subtract_dispersion = self.params_fft.child('Subtract dispersion').value()

        # Calculate background spectrograms------------------------------------------------

        for side in self.spect_params:
            for band in self.spect_params[side]:
                nperseg = self.spect_params[side][band]['nperseg']
                noverlap = self.spect_params[side][band]['noverlap']
                nfft = self.spect_params[side][band]['nfft']
                burst_size = self.burst_size
                sweep = burst_size // 2
                subtract_dispersion = self.spect_params[side][band]['subtract dispersion']

                self.background_spectrograms[side][band] = self.calculate_spectrogram(band, side, nperseg, noverlap, nfft, sweep, burst_size, subtract_dispersion)[5]

        # Plot everything-------------------------------------------------------------------

        self.update_plot()
        self.update_fft()
        self.update_all_beatf()
        self.draw_beatf()
        self.update_profile()

        # Set limits to the parameters-----------------------------------------------------

        self.time_stamps = rpspy.get_timestamps(self.shot, self.file_path)

        #TODO: Handle cases with different number of sweeps and different number of points per sweep
        self.params_sweep.child('Sweep').setLimits((1, len(self.time_stamps)))
        self.params_sweep.child('Sweep nº').setLimits((1, len(self.time_stamps)))
        self.params_sweep.child('Timestamp').setLimits((self.time_stamps[0], self.time_stamps[-1]))
        self.params_sweep.child('Timestamp').setOpts(step=self.time_stamps[1])
        self.params_fft.child('nperseg').setLimits((MIN_NPERSEG, len(self.data)))
        self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
        self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), MAX_NFFT))
        self.params_filter.child('Low Filter').setLimits((0, np.inf))
        self.params_filter.child('High Filter').setLimits((abs(self.f_beat[0] - self.f_beat[1]), np.inf))
        self.params_reconstruct.child('Start Time').setLimits((0, len(self.time_stamps) * (self.time_stamps[1] - self.time_stamps[0])))
        self.params_reconstruct.child('End Time').setLimits((0, len(self.time_stamps) * (self.time_stamps[1] - self.time_stamps[0])))


    def update_plot(self):
        start_time = time.time()

        if self.band == 'V':
            self.signal_type = 'complex'
        else:
            self.signal_type = 'real'
            
        self.data = rpspy.get_band_signal(self.shot, self.file_path, self.band, self.side, self.signal_type, self.sweep)[0]
        self.data -= 2**11 if self.signal_type == 'real' else (2**11 + 1j * 2**11)

        if SANDBOX:
            self.x_data = func_aux.cached_get_linearization(self.shot, 24, self.band, shotfile_dir=self.file_path)
        else:
            self.x_data = func_aux.cached_get_auto_linearization_from_shares(self.shot, self.band)
        self.x_data, self.data = rpspy.linearize(self.x_data, self.data)

        self.draw_plot()

        print("plot")
        print("--- %s seconds ---" % (time.time() - start_time))


    def draw_plot(self):
        y_real = np.real(self.data)
        y_complex = np.imag(self.data)

        self.plot_sweep.clear()
        self.plot_sweep.plot(self.x_data, y_real, pen=pg.mkPen(color='r', width=2)) #Plot real part

        if not(np.array_equiv(y_complex, 0)):
            self.plot_sweep.plot(self.x_data, y_complex, pen=pg.mkPen(color='b', width=2)) #Plot complex part
        
        self.plot_sweep.setLimits(xMin=self.x_data[0],
                                xMax=self.x_data[-1],
                                yMin=-2**11,
                                yMax=2**11)
        
        """ self.plot_sweep.setRange(xRange=(self.x_data[0], self.x_data[-1]),
                                    yRange=(0, 2**12)) """
        
        self.plot_sweep.setLabel('bottom', 'Probing Frequency', units='Hz')


    def update_fft(self):
        start_time = time.time()

        self.f, self.fs, self.f_beat, self.t, self.f_probe, self.unfiltered_Sxx = self.calculate_spectrogram(self.band, self.side, self.nperseg, self.noverlap, self.nfft, self.sweep, self.burst_size, self.subtract_dispersion)
        
        self.draw_spectrogram()

        print("fft")
        print("--- %s seconds ---" % (time.time() - start_time))


    def draw_spectrogram(self):
        if self.subtract_background:
            self.Sxx = self.background_subtract(self.unfiltered_Sxx, self.band, self.side)
        else:
            self.Sxx = np.array(self.unfiltered_Sxx)

        if self.params_fft.child('Scale').value() == 'Normalized':
            Sxx_copy = np.array(self.Sxx)
            max_vals = np.max(Sxx_copy, axis=0)
            max_vals[max_vals == 0] = 1  # Prevent division by zero
            Sxx_copy = Sxx_copy / max_vals

        elif self.params_fft.child('Scale').value() == 'Linear':
            Sxx_copy = np.array(self.Sxx)
        
        elif self.params_fft.child('Scale').value() == 'Logarithmic':
            Sxx_copy = np.log(np.array(self.Sxx))
        
        # Transformed display of ImageItem
        transform = QtGui.QTransform() # prepare ImageItem transformation
        alpha_x = (self.nperseg-self.noverlap)*abs(self.f[1]-self.f[0])
        transform.translate(self.noverlap/2*abs(self.f[1]-self.f[0]) + self.f[0], -self.fs/2 if self.band == 'V' else 0)
        transform.scale(alpha_x, abs(self.f_beat[1]-self.f_beat[0])) # scale horizontal and vertical axes

        i1 = pg.ImageItem(image=Sxx_copy.T) # Note: `Sxx` needs to be transposed to fit the display format
        i1.setTransform(transform) # assign transform

        # Check if Sxx has nan values
        if np.isnan(Sxx_copy).any():
            print("Warning: Sxx has NaN values")
        
        self.plot_spect.clear() # Clear previous plot
        self.plot_spect.addItem(i1)
        
        # Set up color bar
        colormap = self.params_fft.child('Color Map').value()
        try:
            self.colorBar.setImageItem(i1)
            self.colorBar.setColorMap(colormap)
            self.colorBar.setLevels(values=(np.min(Sxx_copy), np.max(Sxx_copy)))
        except AttributeError:
            self.colorBar = self.plot_spect.addColorBar(i1, colorMap=colormap, values=(np.min(Sxx_copy), np.max(Sxx_copy)))

        # Configure plot appearance
        self.plot_spect.setMouseEnabled(x=True, y=True)
        self.plot_spect.setLimits(xMin=self.f_probe[0]-(self.f_probe[1]-self.f_probe[0])/2,
                                xMax=self.f_probe[-1]+(self.f_probe[1]-self.f_probe[0])/2,
                                yMin=self.f_beat[0]-(self.f_beat[1]-self.f_beat[0])/2,
                                yMax=self.f_beat[-1]+(self.f_beat[1]-self.f_beat[0])/2)
        
        """ self.plot_spect.setRange(xRange=(self.f_probe[0]-(self.f_probe[1]-self.f_probe[0])/2, self.f_probe[-1]+(self.f_probe[1]-self.f_probe[0])/2),
                               yRange=(self.f_beat[0]-(self.f_beat[1]-self.f_beat[0])/2, self.f_beat[-1]+(self.f_beat[1]-self.f_beat[0])/2)) """
        
        self.plot_spect.setLabel('bottom', 'Probing Frequency', units='Hz')
        self.plot_spect.setLabel('left', 'Beat Frequency', units='Hz')

        self.draw_dispersion_line()

        self.draw_low_filter()

        self.draw_high_filter()

        self.draw_beatf_spectrogram()

        
    def draw_dispersion_line(self):
        self.y_dis = self.calculate_dispersion(self.band, self.side, self.f_probe, self.t, self.subtract_dispersion)
        self.plot_spect.plot(self.f_probe, self.y_dis, pen=pg.mkPen(color='g', width=2))


    def draw_low_filter(self):
        filter_low = self.filters[self.side][self.band][0]
        y_low = self.y_dis + filter_low
        self.plot_spect.plot(self.f_probe, y_low, pen=pg.mkPen(color='b', width=2))


    def draw_high_filter(self):
        filter_high = self.filters[self.side][self.band][1]
        y_high = self.y_dis + filter_high
        self.plot_spect.plot(self.f_probe, y_high, pen=pg.mkPen(color='w', width=2))


    def draw_beatf_spectrogram(self):
        Sxx_copy = np.array(self.Sxx)
        self.y_beatf = self.calculate_beatf(self.band, self.side, Sxx_copy, self.y_dis, self.f_beat, self.fs)
        self.plot_spect.plot(self.f_probe, self.y_beatf, pen=pg.mkPen(color='r', width=2))


    def update_one_beatf(self, band, side):
        if side == self.side and band == self.band:
            self.df_dt = (self.f[-1] - self.f[0])/((len(self.f) - 1)*(1/self.fs))
            
            y_beat_time = (self.y_beatf - self.y_dis) / self.df_dt

            self.beat_frequencies[side][band] = [self.f_probe, self.y_beatf, y_beat_time, self.df_dt]
        
        else:
            nperseg = self.spect_params[side][band]['nperseg']
            noverlap = self.spect_params[side][band]['noverlap']
            nfft = self.spect_params[side][band]['nfft']
            burst_size = self.burst_size
            subtract_dispersion = self.spect_params[side][band]['subtract dispersion']
            subtract_background = self.spect_params[side][band]['subtract background']

            f, fs, f_beat, t, f_probe, Sxx = self.calculate_spectrogram(band, side, nperseg, noverlap, nfft, self.sweep, burst_size, subtract_dispersion)

            if subtract_background:
                Sxx = self.background_subtract(Sxx, band, side)

            y_dis = self.calculate_dispersion(band, side, f_probe, t, subtract_dispersion)

            y_beatf = self.calculate_beatf(band, side, Sxx, y_dis, f_beat, fs)

            df_dt = (f[-1] - f[0])/((len(f) - 1)*(1/fs))
            
            y_beat_time = (y_beatf - y_dis) / df_dt

            self.beat_frequencies[side][band] = [f_probe, y_beatf, y_beat_time, df_dt]


    def update_all_beatf(self):
        start_time = time.time()

        for side in self.spect_params:
            for band in self.spect_params[side]:
                self.update_one_beatf(band, side)

        print("beatf's")
        print("--- %s seconds ---" % (time.time() - start_time))


    def draw_beatf(self):
        self.plot_beatf.clear()

        self.all_delay_HFS_f_probe = np.concatenate((
            [0],
            self.beat_frequencies['HFS']['K'][0],
            self.beat_frequencies['HFS']['Ka'][0],
            self.beat_frequencies['HFS']['Q'][0],
            self.beat_frequencies['HFS']['V'][0]
        )
        )


        self.all_delay_HFS_beat_time = np.concatenate((
            [0],
            self.beat_frequencies['HFS']['K'][2],
            self.beat_frequencies['HFS']['Ka'][2],
            self.beat_frequencies['HFS']['Q'][2],
            self.beat_frequencies['HFS']['V'][2]
        )
        )

        self.all_delay_LFS_f_probe = np.concatenate((
            [0],
            self.beat_frequencies['LFS']['K'][0],
            self.beat_frequencies['LFS']['Ka'][0],
            self.beat_frequencies['LFS']['Q'][0],
            self.beat_frequencies['LFS']['V'][0]
        )
        )

        self.all_delay_LFS_beat_time = np.concatenate(
            (
            [0],
            self.beat_frequencies['LFS']['K'][2],
            self.beat_frequencies['LFS']['Ka'][2],
            self.beat_frequencies['LFS']['Q'][2],
            self.beat_frequencies['LFS']['V'][2]
        )
        )
        
        # Handle NaNs in these arrays
        # HFS
        nan_mask = np.isnan(self.all_delay_HFS_beat_time)
        self.all_delay_HFS_beat_time = self.all_delay_HFS_beat_time[~nan_mask]
        self.all_delay_HFS_f_probe = self.all_delay_HFS_f_probe[~nan_mask]

        # LFS
        nan_mask = np.isnan(self.all_delay_LFS_beat_time)
        self.all_delay_LFS_beat_time = self.all_delay_LFS_beat_time[~nan_mask]
        self.all_delay_LFS_f_probe = self.all_delay_LFS_f_probe[~nan_mask]

        self.plot_beatf.plot(self.all_delay_HFS_f_probe, self.all_delay_HFS_beat_time, pen=pg.mkPen(color='w', width=2))
        self.plot_beatf.plot(self.all_delay_LFS_f_probe, self.all_delay_LFS_beat_time, pen=pg.mkPen(color='w', width=2))

        for side in self.beat_frequencies:
            for band in self.beat_frequencies[side]:
                f_probe = self.beat_frequencies[side][band][0]
                beat_time = self.beat_frequencies[side][band][2]
                #beat_freq = self.beat_frequencies[side][band][1]
                self.plot_beatf.plot(f_probe, beat_time, pen=pg.mkPen(color='r' if side == 'HFS' else 'b', width=2))

        self.plot_beatf.setLabel('bottom', 'Probing Frequency', units='Hz')
        self.plot_beatf.setLabel('left', 'Time Delay', units='s')


    def update_profile(self):
        start_time = time.time()

        group_delay_HFS_x = np.linspace(self.all_delay_HFS_f_probe[0], self.all_delay_HFS_f_probe[-1], PROFILE_INVERSION_RESOLUTION)
        group_delay_HFS_y = np.interp(group_delay_HFS_x, self.all_delay_HFS_f_probe, self.all_delay_HFS_beat_time)

        group_delay_LFS_x = np.linspace(self.all_delay_LFS_f_probe[0], self.all_delay_LFS_f_probe[-1], PROFILE_INVERSION_RESOLUTION)
        group_delay_LFS_y = np.interp(group_delay_LFS_x, self.all_delay_LFS_f_probe, self.all_delay_LFS_beat_time)
        
        r_HFS = rpspy.profile_inversion(group_delay_HFS_x, group_delay_HFS_y, pwld_batch=True)
        r_LFS = rpspy.profile_inversion(group_delay_LFS_x, group_delay_LFS_y, pwld_batch=True)
        
        ne_HFS = rpspy.f_to_ne(group_delay_HFS_x)
        ne_LFS = rpspy.f_to_ne(group_delay_LFS_x)

        self.plot_profile.clear()
        self.plot_profile.plot(r_HFS, ne_HFS*1e-19, pen=pg.mkPen(color='r', width=2))
        self.plot_profile.plot(r_LFS, ne_LFS*1e-19, pen=pg.mkPen(color='r', width=2))
        # self.plot_profile.setLimits(xMin=min(x), 
        #                             xMax=max(x), 
        #                             maxXRange=max(x)-min(x), 
        #                             yMin=min(self.density*1e-19),
        #                             yMax=max(self.density*1e-19),
        #                             maxYRange=max(self.density*1e-19)-min(self.density*1e-19))
        self.plot_profile.setLabel('bottom', 'radius', units='m')
        self.plot_profile.setLabel('left', 'density', units='1e19 m^-3')

        print("profile")
        print("--- %s seconds ---" % (time.time() - start_time))
    
# Calculate----------------------------------------------------------------------------------------------------------------------
    
    def calculate_spectrogram(self, band, side, nperseg, noverlap, nfft, sweep, burst_size, subtract=None):
        if band == 'V':
            signal_type = 'complex'
        else:
            signal_type = 'real'

        burst = rpspy.get_band_signal(self.shot, self.file_path, band, side, signal_type, sweep - burst_size // 2, burst_size)
        # f = func_aux.cached_get_linearization(self.shot, 24, band, shotfile_dir=self.file_path)
        if SANDBOX:
            f = func_aux.cached_get_linearization(self.shot, 24, band, shotfile_dir=self.file_path)
        else:
            f = func_aux.cached_get_auto_linearization_from_shares(self.shot, band)
        f, linearized_burst = rpspy.linearize(f, burst)

        fs = rpspy.get_sampling_frequency(self.shot, self.file_path)

        if band == 'V':
            linearized_burst -= (2**11 + 1j * 2**11)
        else:
            linearized_burst -= 2**11

        if subtract == True:
            df_dt = (f[-1] - f[0])/((len(f) - 1)*(1/fs))
            correction = rpspy.get_dispersion_phase(self.shot, band, side, f[0], df_dt, np.arange(len(f))/fs)
            corrected_burst = linearized_burst * correction  # Multiply by dispersion phase
        else:
            corrected_burst = linearized_burst

        f_beat, t, Sxx = spectrogram(
            corrected_burst, 
            fs=fs, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            nfft=nfft,
            return_onesided=False if corrected_burst.dtype == complex else True,
            detrend=False
            )

        f_probe = np.interp(t, np.arange(len(f))/fs, f)

        if band == 'V':
            f_beat = fftshift(f_beat)
            Sxx = fftshift(Sxx, axes=-2)

        # Calculate average of the burst
        Sxx = np.average(Sxx, axis=0)

        return f, fs, f_beat, t, f_probe, Sxx


    def calculate_dispersion(self, band, side, f_probe, t, subtract):
        if band == 'V' and subtract == True:
            y_dis = np.zeros(len(f_probe))
        else:
            k = (f_probe[-1] - f_probe[0]) / (t[-1] - t[0])
            y_dis = k * rpspy.aug_tgcorr2(band, side, f_probe*1e-9, self.shot)

        return y_dis


    def calculate_beatf(self, band, side, Sxx, y_dis, f_beat, fs):
        filter_low = self.filters[side][band][0]
        filter_high = self.filters[side][band][1]

        # Apply filters to spectrogram
        Sxx[np.broadcast_to(f_beat[:, None], Sxx.shape) <= y_dis + filter_low] = Sxx.min()
        Sxx[np.broadcast_to(f_beat[:, None], Sxx.shape) >= y_dis + filter_high] = Sxx.min()
        
        # Generate the line through the max of the graph
        y_max, _ = rpspy.column_wise_max_with_quadratic_interpolation(Sxx)  # Y coordinates
        y_max *= abs(f_beat[1]-f_beat[0])

        if band == 'V':
            y_max += -fs/2
        
        return y_max
    

    def background_subtract(self, Sxx, band, side):
        spectrogram_subtracted = np.clip(
            Sxx - self.background_spectrograms[side][band],
            a_min=np.min(Sxx),
            a_max=np.inf,
        )

        return spectrogram_subtracted

# Parameters ---------------------------------------------------------------------------------------------------------------------

    def save_config(self):
        path = self.params_config.child('Save').value()

        data = {'parameters': self.spect_params,
                'filters': self.filters,
                'burst_size': self.burst_size}
        
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
        
        print(f"Data saved to {path}")

        # Workaround to force sigValueChanged and overwright existing file
        self.params_config.child('Save').setValue(path + ' (saved)', blockSignal=self.save_config)
        self.params_config.child('Load').setValue('', blockSignal=self.load_config)


    def load_config(self):
        path = self.params_config.child('Load').value()
        
        with open(path, 'r') as file:
            # Parse JSON content
            data = json.load(file)

            # Extract data from the parsed JSON
            self.spect_params = data.get("parameters", {})
            self.filters = data.get("filters", {})
            self.burst_size = data.get("burst_size", 0.0)

        self.set_saved_params()

        # Force change signal to handle sweep number and plot everything
        self.params_fft.child('burst size (odd)').setValue(0, blockSignal=self.update_fft_params)
        self.params_fft.child('burst size (odd)').setValue(self.burst_size)

        # Workaround to force sigValueChanged and load from overwritten file
        self.params_config.child('Load').setValue(path + ' (loaded)', blockSignal=self.load_config)
        self.params_config.child('Save').setValue('', blockSignal=self.save_config)


    def set_saved_params(self):
        self.params_filter.child('Low Filter').setValue(self.filters[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()][0], blockSignal=self.update_fft_params)
        self.params_filter.child('High Filter').setValue(self.filters[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()][1], blockSignal=self.update_fft_params)
        self.params_fft.child('nperseg').setValue(self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['nperseg'], blockSignal=self.update_fft_params)
        self.nperseg = self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['nperseg']
        self.params_fft.child('noverlap').sigValueChanged.disconnect(self.update_fft_params)
        self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
        self.params_fft.child('noverlap').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('noverlap').setValue(self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['noverlap'], blockSignal=self.update_fft_params)
        self.noverlap = self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['noverlap']
        self.params_fft.child('nfft').sigValueChanged.disconnect(self.update_fft_params)
        self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), np.inf))
        self.params_fft.child('nfft').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('nfft').setValue(self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['nfft'], blockSignal=self.update_fft_params)
        self.nfft = self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['nfft']
        self.params_fft.child('Subtract background').setValue(self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['subtract background'], blockSignal=self.update_fft_params)
        self.subtract_background = self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['subtract background']
        self.params_fft.child('Subtract dispersion').setValue(self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['subtract dispersion'], blockSignal=self.update_fft_params)
        self.subtract_dispersion = self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['subtract dispersion']


    def update_detector_params(self):
        sender = self.sender()

        self.set_saved_params()

        self.band = self.params_detector.child('Band').value()
        self.side = self.params_detector.child('Side').value()

        if sender == self.params_detector.child('Band'):
            if sender.value() == 'V':
                self.params_fft.child('Subtract dispersion').setOpts(enabled=True)
            else:
                self.params_fft.child('Subtract dispersion').setOpts(enabled=False)

        self.update_plot()
        self.update_fft()


    def update_plot_params(self):
        sender = self.sender()

        if sender == self.params_sweep.child('Sweep'):
            value = self.params_sweep.child('Sweep').value()
            self.sweep = value - 1
            timestamp = self.time_stamps[value - 1]
            self.params_sweep.child('Sweep nº').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)

        elif sender == self.params_sweep.child('Sweep nº'):
            value = int(self.params_sweep.child('Sweep nº').value())
            self.sweep = value - 1
            timestamp = self.time_stamps[value - 1]
            self.params_sweep.child('Sweep nº').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)
            
        elif sender == self.params_sweep.child('Timestamp'):
            value = self.params_sweep.child('Timestamp').value()
            timestamp = func_aux.round_to_nearest(value, self.time_stamps)
            index = np.where(self.time_stamps == timestamp)
            self.sweep = index[0][0]
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep').setValue(index[0][0] + 1, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep nº').setValue(index[0][0] + 1, blockSignal=self.update_plot_params)
        
        self.update_plot()

        if not self.supress_updates_ffts:
            self.update_fft()
            self.update_all_beatf()
            self.draw_beatf()
            self.update_profile()


    def update_fft_params(self):
        sender = self.sender()

        if sender == self.params_fft.child('nperseg'):
            value = int(self.params_fft.child('nperseg').value())
            self.spect_params[self.side][self.band]['nperseg'] = value
            self.nperseg = value
            self.params_fft.child('nperseg').setValue(value, blockSignal=self.update_fft_params)
            self.supress_updates_ffts = True
            self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
            self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), np.inf))
            self.supress_updates_ffts = False
        
        elif sender == self.params_fft.child('noverlap'):
            value = int(self.params_fft.child('noverlap').value())
            self.spect_params[self.side][self.band]['noverlap'] = value
            self.noverlap = value
            self.params_fft.child('noverlap').setValue(value, blockSignal=self.update_fft_params)
        
        elif sender == self.params_fft.child('nfft'):
            value = int(self.params_fft.child('nfft').value())
            self.spect_params[self.side][self.band]['nfft'] = value
            self.nfft = value
            self.params_fft.child('nfft').setValue(value, blockSignal=self.update_fft_params)
        
        elif sender == self.params_fft.child('burst size (odd)'):
            value = int(self.params_fft.child('burst size (odd)').value())
            if value % 2 == 0:
                self.params_fft.child('burst size (odd)').setValue(value - 1, blockSignal=self.update_fft_params)
                self.burst_size = value - 1
            else:
                self.params_fft.child('burst size (odd)').setValue(value, blockSignal=self.update_fft_params)
                self.burst_size = value
            lower_limit = int(1 + self.params_fft.child('burst size (odd)').value() // 2)
            upper_limit = int(len(self.time_stamps) - self.params_fft.child('burst size (odd)').value() // 2)
            self.supress_updates_ffts = True
            self.params_sweep.child('Sweep').setLimits([lower_limit, upper_limit])
            self.params_sweep.child('Sweep nº').setLimits([lower_limit, upper_limit])
            self.params_sweep.child('Timestamp').setLimits([self.time_stamps[lower_limit - 1], self.time_stamps[upper_limit - 1]])
            self.supress_updates_ffts = False
        
        elif sender == self.params_fft.child('Subtract background'):
            self.spect_params[self.side][self.band]['subtract background'] = self.params_fft.child('Subtract background').value()
            self.subtract_background = self.params_fft.child('Subtract background').value()

        elif sender == self.params_fft.child('Subtract dispersion'):
            self.spect_params[self.side][self.band]['subtract dispersion'] = self.params_fft.child('Subtract dispersion').value()
            self.subtract_dispersion = self.params_fft.child('Subtract dispersion').value()

        elif sender == self.params_filter.child('Low Filter'):
            self.filters[self.side][self.band][0] = self.params_filter.child('Low Filter').value()
            if self.filters[self.side][self.band][0] + abs(self.f_beat[1] - self.f_beat[0]) >= self.filters[self.side][self.band][1]:
                self.filters[self.side][self.band][1] = self.filters[self.side][self.band][0] + abs(self.f_beat[1] - self.f_beat[0])
                self.params_filter.child('High Filter').setValue(self.filters[self.side][self.band][1], blockSignal=self.update_fft_params)
        
        elif sender == self.params_filter.child('High Filter'):
            self.filters[self.side][self.band][1] = self.params_filter.child('High Filter').value()
            if self.filters[self.side][self.band][1] - abs(self.f_beat[1] - self.f_beat[0]) <= self.filters[self.side][self.band][0]:
                self.filters[self.side][self.band][0] = self.filters[self.side][self.band][1] - abs(self.f_beat[1] - self.f_beat[0])
                self.params_filter.child('Low Filter').setValue(self.filters[self.side][self.band][0], blockSignal=self.update_fft_params)

        if not self.supress_updates_ffts:
            if sender == self.params_filter.child('Low Filter') or sender == self.params_filter.child('High Filter') or sender == self.params_fft.child('Subtract background'):
                self.draw_spectrogram()
            else:
                self.background_spectrograms[self.side][self.band] = self.calculate_spectrogram(self.band, self.side, self.nperseg, self.noverlap, self.nfft, self.burst_size // 2, self.burst_size, self.subtract_dispersion)[5]
                self.update_fft()

            if sender == self.params_fft.child('burst size (odd)'):
                self.update_all_beatf()
            else:
                self.update_one_beatf(self.band, self.side)
            
            self.draw_beatf()
            self.update_profile()
    

    def update_reconstruct_params(self):
        sender = self.sender()

        if sender == self.params_reconstruct.child('Start Time'):
            if self.params_reconstruct.child('Start Time').value() > self.params_reconstruct.child('End Time').value():
                self.params_reconstruct.child('End Time').setValue(self.params_reconstruct.child('Start Time').value(), blockSignal=self.update_reconstruct_params)

        elif sender == self.params_reconstruct.child('End Time'):
            if self.params_reconstruct.child('End Time').value() < self.params_reconstruct.child('Start Time').value():
                self.params_reconstruct.child('Start Time').setValue(self.params_reconstruct.child('End Time').value(), blockSignal=self.update_reconstruct_params)


    @pyqtSlot()
    def request_reconstruct(self):
        self.request_signal.emit()
        self.params_reconstruct.child('Start Time').setOpts(enabled=False)
        self.params_reconstruct.child('End Time').setOpts(enabled=False)
        self.params_reconstruct.child('Time Step').setOpts(enabled=False)
        self.params_reconstruct.child('Reconstruct Shot').setOpts(enabled=False)
    

    @pyqtSlot()
    def finished_reconstruct(self):
        self.params_reconstruct.child('Start Time').setOpts(enabled=True)
        self.params_reconstruct.child('End Time').setOpts(enabled=True)
        self.params_reconstruct.child('Time Step').setOpts(enabled=True)
        self.params_reconstruct.child('Reconstruct Shot').setOpts(enabled=True)



class Threaded(QObject):
    finished_signal = pyqtSignal()


    def __init__(self, parent=None, **kwargs):
        # intentionally not setting the parent
        super().__init__(parent=None, **kwargs)


    @pyqtSlot()
    def reconstruct(self):
        rpspy.full_profile_reconstruction(
            shot=main_window.shot, 
            destination_dir = 'reconstruction_shots', 
            shotfile_dir=main_window.file_path, 
            linearization_shotfile_dir=main_window.file_path, 
            sweep_linearization=None, 
            shot_linearization=main_window.shot,
            spectrogram_options={
            'K': {'nperseg': main_window.nperseg, 'noverlap':main_window.noverlap, 'nfft': main_window.nfft},
            'Ka': {'nperseg': main_window.nperseg, 'noverlap':main_window.noverlap, 'nfft': main_window.nfft},
            'Q': {'nperseg': main_window.nperseg, 'noverlap':main_window.noverlap, 'nfft': main_window.nfft},
            'V': {'nperseg': main_window.nperseg, 'noverlap':main_window.noverlap, 'nfft': main_window.nfft},
            }, 
            filters=main_window.filters,
            subtract_on_bands=None,
            start_time = main_window.params_reconstruct.child('Start Time').value(), 
            end_time = main_window.params_reconstruct.child('End Time').value(), 
            time_step = main_window.params_reconstruct.child('Time Step').value(),
            burst = int(main_window.params_fft.child('burst size (odd)').value()), 
            write_dump = True,
            return_profiles = False,
            )
        self.finished_signal.emit()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = PlotWindow()
    main_window.show()
    sys.exit(app.exec())