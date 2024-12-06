import sys
import numpy as np
import json
from scipy.signal import spectrogram
from scipy.fft import fftshift
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QSplitter, QHeaderView
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import rpspy
import func_aux
import time


#TODO: Remove hardcoded values and add them here
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

# Param Tree
DEFAULT_SECTION_SIZE = 200

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
        self.data = None
        self.burst = None
        self.nperseg = None
        self.noverlap = None
        self.nfft = None
        self.colormap = None
        self.Sxx = None
        self.subtract = None
        self.params_added = False
        self.supress_updates = False #Prevents repeated call of the update functions
        
        #Store the spectrogram parameters
        self.spect_params = {
            'HFS': {
                'K': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract': None
                },
                'Ka': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract': None
                },
                'Q': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract': None
                },
                'V': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract': True
                }
            },
            'LFS': {
                'K': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract': None
                },
                'Ka': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract': None
                },
                'Q': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract': None
                },
                'V': {
                    'nperseg': DEFAULT_NPERSEG,
                    'noverlap': DEFAULT_NOVERLAP,
                    'nfft': DEFAULT_NFFT,
                    'subtract': True
                }
            }
        }

        #Store the beat frquencies of the 8 detectors
        self.beat_frequencies = {
            'HFS': {
                'K': [None, None, None, None], # [[f_probe], [beatf], [beat_time], [df/dt]]
                'Ka': [None, None, None, None],
                'Q': [None, None, None, None],
                'V': [None, None, None, None]
            },
            'LFS': {
                'K': [None, None, None, None],
                'Ka': [None, None, None, None],
                'Q': [None, None, None, None],
                'V': [None, None, None, None]
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

        #---------------------------------------------------------------------------

        # Set the application icon
        self.setWindowIcon(QtGui.QIcon('reflecto-lab.png'))

        # Set up the main window
        self.setWindowTitle('ReflectoLab')
        self.setGeometry(100, 100, 1600, 800)
        
        # Create layouts and widgets------------------------------------------------

        # Create a central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        # Create a QSplitter for adjustable layout
        self.splitter = QSplitter()
        
        # Create a second QSplitter for adjustable graphs
        self.splitter_graphs = QSplitter()

        # Create the main graphics layout widget
        self.graph_layout1 = pg.GraphicsLayoutWidget()
        self.graph_layout2 = pg.GraphicsLayoutWidget()
        
        # Create the parameter tree
        self.params_file = Parameter.create(name='File', type='group', children=[
            {'name': 'Open', 'type': 'file', 'value': None, 'fileMode': 'Directory'},
            {'name': 'Shot', 'type': 'int'}
        ])
        self.params_detector = Parameter.create(name='Detector', type='group', children=[
            {'name': 'Band', 'type': 'list', 'limits': ['K', 'Ka', 'Q', 'V']},
            {'name': 'Side', 'type': 'list', 'limits': ['HFS', 'LFS']}
        ])
        self.params_sweep = Parameter.create(name='Sweep', type='group', children=[
            {'name': 'Sweep', 'type': 'slider', 'limits': (1, 1)},
            {'name': 'Sweep nº', 'type': 'float', 'value': 1, 'decimals': DECIMALS_SWEEP_NUM, 'delay': 0},
            {'name': 'Timestamp', 'type': 'float', 'value': 0, 'suffix': 's', 'decimals': DECIMALS_TIMESTAMP, 'siPrefix': True, 'delay': 0},
        ])
        self.params_fft = Parameter.create(name='Spectrogram', type='group', children=[
            {'name': 'nperseg', 'type': 'float', 'value': DEFAULT_NPERSEG},
            {'name': 'noverlap', 'type': 'float', 'value': DEFAULT_NOVERLAP},
            {'name': 'nfft', 'type': 'float', 'value': DEFAULT_NFFT},
            {'name': 'burst size (odd)', 'type': 'float', 'value': DEFAULT_BURST_SIZE, 'limits': (1, MAX_BURST_SIZE), 'step': 2, 'delay': 0},
            {'name': 'Scale', 'type': 'checklist', 'limits': ['Normalized', 'Linear', 'Logarithmic'], 'exclusive': True, 'delay': 0},
            {'name': 'Subtract dispersion', 'type': 'bool', 'value': False, 'enabled': False},
            {'name': 'Color Map', 'type': 'cmaplut', 'value': 'plasma'}
        ])
        self.params_filter = Parameter.create(name='Filters (above dispersion)', type='group', children=[
            {'name': 'Low Filter', 'type': 'float', 'value': DEFAULT_FILTER_LOW, 'suffix': 'Hz', 'siPrefix': True},
            {'name': 'High Filter', 'type': 'float', 'value': DEFAULT_FILTER_HIGH, 'suffix': 'Hz', 'siPrefix': True}
        ])
        self.params_reconstruct = Parameter.create(name='Reconstruct Shot', type='group', children=[
            {'name': 'Start Time', 'type': 'float', 'value': DEFAULT_START_TIME, 'suffix': 's', 'siPrefix': True},
            {'name': 'End Time', 'type': 'float', 'value': DEFAULT_END_TIME, 'suffix': 's', 'siPrefix': True},
            {'name': 'Time Step', 'type': 'float', 'value': DEFAULT_TIMESTEP, 'suffix': 's', 'siPrefix': True},
            {'name': 'Reconstruct Shot', 'type': 'action'}
        ])
        self.param_tree = ParameterTree()
        self.param_tree.header().setDefaultSectionSize(DEFAULT_SECTION_SIZE)
        self.param_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.param_tree.addParameters(self.params_file)

        # Create the first and second plots
        self.plot_sweep = self.graph_layout1.addPlot(title="Sweep")
        self.graph_layout1.nextRow()
        self.plot_fft = self.graph_layout1.addPlot(title="Spectrogram")

        self.plot_profile = self.graph_layout2.addPlot(title="Profile")
        self.graph_layout2.nextRow()
        self.plot_beatf = self.graph_layout2.addPlot(title="Beat Frequency")

        # Add widgets and layouts---------------------------------------------------

        # Add widgets to the splitter
        self.splitter.addWidget(self.param_tree)
        self.splitter_graphs.addWidget(self.graph_layout1)
        self.splitter_graphs.addWidget(self.graph_layout2)
        self.splitter.addWidget(self.splitter_graphs)

        self.splitter.setSizes([400, 1200])
        
        # Add the splitter to the main layout
        self.layout.addWidget(self.splitter)

        #---------------------------------------------------------------------------

        # Connect the file to the function that displays the rest of the parameters and the graphs
        self.params_file.child('Open').sigValueChanged.connect(self.update_shot)
        self.params_file.child('Shot').sigValueChanged.connect(self.update_shot)
    

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
        
        
        if self.params_added == False:
            self.param_tree.addParameters(self.params_detector)
            self.param_tree.addParameters(self.params_sweep)
            self.param_tree.addParameters(self.params_fft)
            self.param_tree.addParameters(self.params_filter)
            self.param_tree.addParameters(self.params_reconstruct)
            self.params_added = True

        # Connect the parameters to the functions-----------------------------------

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

        #---------------------------------------------------------------------------

        self.update_plot()
        self.update_fft()
        self.update_all_beatf()
        self.update_profile()

        # Set limits to the parameters----------------------------------------------

        self.params_sweep.child('Sweep').setLimits((1, len(rpspy.get_timestamps(self.shot, self.file_path))))
        self.params_sweep.child('Sweep nº').setLimits((1, len(rpspy.get_timestamps(self.shot, self.file_path))))
        self.params_sweep.child('Timestamp').setLimits((rpspy.get_timestamps(self.shot, self.file_path)[0], rpspy.get_timestamps(self.shot, self.file_path)[-1]))
        self.params_sweep.child('Timestamp').setOpts(step=rpspy.get_timestamps(self.shot, self.file_path)[1])
        self.params_fft.child('nperseg').setLimits((MIN_NPERSEG, len(self.data)))
        self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
        self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), MAX_NFFT))
        self.params_filter.child('Low Filter').setLimits((0, np.inf))
        self.params_filter.child('High Filter').setLimits((abs(self.f_beat[0] - self.f_beat[1]), np.inf))
        self.params_reconstruct.child('Start Time').setLimits((0, len(rpspy.get_timestamps(self.shot, self.file_path)) * (rpspy.get_timestamps(self.shot, self.file_path)[1] - rpspy.get_timestamps(self.shot, self.file_path)[0])))
        self.params_reconstruct.child('End Time').setLimits((0, len(rpspy.get_timestamps(self.shot, self.file_path)) * (rpspy.get_timestamps(self.shot, self.file_path)[1] - rpspy.get_timestamps(self.shot, self.file_path)[0])))
        

    def update_plot(self):
        start_time = time.time()

        self.band = self.params_detector.child('Band').value()
        self.side = self.params_detector.child('Side').value()
        self.sweep = int(self.params_sweep.child('Sweep nº').value()) - 1

        if self.band == 'V':
            self.signal_type = 'complex'
        else:
            self.signal_type = 'real'
            
        self.data = rpspy.get_band_signal(self.shot, self.file_path, self.band, self.side, self.signal_type, self.sweep)[0]
        self.x_data = func_aux.cached_get_linearization(self.shot, 24, self.band, shotfile_dir=self.file_path)
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
                                yMin=0,
                                yMax=2**12)
        
        """ self.plot_sweep.setRange(xRange=(self.x_data[0], self.x_data[-1]),
                                    yRange=(0, 2**12)) """
        
        self.plot_sweep.setLabel('bottom', 'Probing Frequency', units='Hz')


    def update_fft(self):
        start_time = time.time()

        self.nperseg = int(self.params_fft.child('nperseg').value())
        self.noverlap = int(self.params_fft.child('noverlap').value())
        self.nfft = int(self.params_fft.child('nfft').value())
        self.burst_size = int(self.params_fft.child('burst size (odd)').value())
        self.subtract = self.params_fft.child('Subtract dispersion').value()
        
        self.f, self.fs, self.f_beat, self.t, self.f_probe, self.Sxx = self.calculate_spectrogram(self.band, self.side, self.nperseg, self.noverlap, self.nfft, self.burst_size, self.subtract)
        
        self.draw_spectrogram()

        print("fft")
        print("--- %s seconds ---" % (time.time() - start_time))


    def draw_spectrogram(self):
        #TODO: Fix scale of graphs when changing sweeps
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
        
        self.plot_fft.clear() # Clear previous plot
        self.plot_fft.addItem(i1)
        
        # Set up color bar
        colormap = self.params_fft.child('Color Map').value()
        try:
            self.colorBar.setImageItem(i1)
            self.colorBar.setColorMap(colormap)
            self.colorBar.setLevels(values=(np.min(Sxx_copy), np.max(Sxx_copy)))
        except AttributeError:
            self.colorBar = self.plot_fft.addColorBar(i1, colorMap=colormap, values=(np.min(Sxx_copy), np.max(Sxx_copy)))

        # Configure plot appearance
        self.plot_fft.setMouseEnabled(x=True, y=True)
        self.plot_fft.setLimits(xMin=self.f_probe[0]-(self.f_probe[1]-self.f_probe[0])/2,
                                xMax=self.f_probe[-1]+(self.f_probe[1]-self.f_probe[0])/2,
                                yMin=self.f_beat[0]-(self.f_beat[1]-self.f_beat[0])/2,
                                yMax=self.f_beat[-1]+(self.f_beat[1]-self.f_beat[0])/2)
        
        """ self.plot_fft.setRange(xRange=(self.f_probe[0]-(self.f_probe[1]-self.f_probe[0])/2, self.f_probe[-1]+(self.f_probe[1]-self.f_probe[0])/2),
                               yRange=(self.f_beat[0]-(self.f_beat[1]-self.f_beat[0])/2, self.f_beat[-1]+(self.f_beat[1]-self.f_beat[0])/2)) """
        
        self.plot_fft.setLabel('bottom', 'Probing Frequency', units='Hz')
        self.plot_fft.setLabel('left', 'Beat Frequency', units='Hz')

        self.draw_dispersion_line()

        self.draw_low_filter()

        self.draw_high_filter()

        self.draw_beatf_spectrogram()

        
    def draw_dispersion_line(self):
        self.y_dis = self.calculate_dispersion(self.band, self.side, self.f_probe, self.t, self.subtract)
        self.plot_fft.plot(self.f_probe, self.y_dis, pen=pg.mkPen(color='g', width=2))


    def draw_low_filter(self):
        filter_low = self.filters[self.side][self.band][0]
        y_low = self.y_dis + filter_low
        self.plot_fft.plot(self.f_probe, y_low, pen=pg.mkPen(color='b', width=2))


    def draw_high_filter(self):
        filter_high = self.filters[self.side][self.band][1]
        y_high = self.y_dis + filter_high
        self.plot_fft.plot(self.f_probe, y_high, pen=pg.mkPen(color='w', width=2))


    def draw_beatf_spectrogram(self):
        Sxx_copy = np.array(self.Sxx)
        self.y_beatf = self.calculate_beatf(self.band, self.side, Sxx_copy, self.y_dis, self.f_beat, self.fs)
        self.plot_fft.plot(self.f_probe, self.y_beatf, pen=pg.mkPen(color='r', width=2))


    def update_all_beatf(self):
        start_time = time.time()

        for side in self.spect_params:
            for band in self.spect_params[side]:
                if side == self.side and band == self.band:
                    
                    if self.beat_frequencies[side][band][3] == None:
                        df_dt = (self.f_probe[-1] - self.f_probe[0])/((len(self.f_probe) - 1)*(1/self.fs))
                    else:
                        df_dt = self.beat_frequencies[side][band][3]
                    
                    y_beat_time = (self.y_beatf - self.y_dis) / df_dt

                    self.beat_frequencies[side][band] = [self.f_probe, self.y_beatf, y_beat_time, df_dt]
                
                else:
                    nperseg = self.spect_params[side][band]['nperseg']
                    noverlap = self.spect_params[side][band]['noverlap']
                    nfft = self.spect_params[side][band]['nfft']
                    burst_size = int(self.params_fft.child('burst size (odd)').value())
                    subtract = self.spect_params[side][band]['subtract']

                    _, fs, f_beat, t, f_probe, Sxx = self.calculate_spectrogram(band, side, nperseg, noverlap, nfft, burst_size, subtract)

                    y_dis = self.calculate_dispersion(band, side, f_probe, t, subtract)

                    y_beatf = self.calculate_beatf(band, side, Sxx, y_dis, f_beat, fs)

                    if self.beat_frequencies[side][band][3] == None:
                        df_dt = (f_probe[-1] - f_probe[0])/((len(f_probe) - 1)*(1/fs))
                    else:
                        df_dt = self.beat_frequencies[side][band][3]
                    
                    y_beat_time = (y_beatf - y_dis) / df_dt

                    self.beat_frequencies[side][band] = [f_probe, y_beatf, y_beat_time, df_dt]
                    
        
        self.draw_beatf()

        print("beatf's")
        print("--- %s seconds ---" % (time.time() - start_time))
    

    def update_one_beatf(self):
        start_time = time.time()

        if self.beat_frequencies[self.side][self.band][3] == None:
            df_dt = (self.f_probe[-1] - self.f_probe[0])/((len(self.f_probe) - 1)*(1/self.fs))
        else:
            df_dt = self.beat_frequencies[self.side][self.band][3]
        
        y_beat_time = (self.y_beatf - self.y_dis) / df_dt

        self.beat_frequencies[self.side][self.band] = [self.f_probe, self.y_beatf, y_beat_time, df_dt]

        self.draw_beatf()
        
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
        
        self.plot_beatf.plot(self.all_delay_HFS_f_probe, self.all_delay_HFS_beat_time, pen=pg.mkPen(color='w', width=2))
        self.plot_beatf.plot(self.all_delay_LFS_f_probe, self.all_delay_LFS_beat_time, pen=pg.mkPen(color='w', width=2))

        for side in self.beat_frequencies:
            for band in self.beat_frequencies[side]:
                f_probe = self.beat_frequencies[side][band][0]
                beat_time = self.beat_frequencies[side][band][2]
                self.plot_beatf.plot(f_probe, beat_time, pen=pg.mkPen(color='r' if side == 'HFS' else 'b', width=2))

        self.plot_beatf.setLabel('bottom', 'Probing Frequency', units='Hz')
        self.plot_beatf.setLabel('left', 'Time Delay', units='s')


    def update_profile(self):
        group_delay_HFS_x = np.linspace(self.all_delay_HFS_f_probe[0], self.all_delay_HFS_f_probe[-1], PROFILE_INVERSION_RESOLUTION)
        group_delay_HFS_y = np.interp(group_delay_HFS_x, self.all_delay_HFS_f_probe, self.all_delay_HFS_beat_time)

        group_delay_LFS_x = np.linspace(self.all_delay_LFS_f_probe[0], self.all_delay_LFS_f_probe[-1], PROFILE_INVERSION_RESOLUTION)
        group_delay_LFS_y = np.interp(group_delay_LFS_x, self.all_delay_LFS_f_probe, self.all_delay_LFS_beat_time)

        start_time = time.time()
        r_HFS = rpspy.profile_inversion(group_delay_HFS_x, group_delay_HFS_y, pwld_batch=True)
        r_LFS = rpspy.profile_inversion(group_delay_LFS_x, group_delay_LFS_y, pwld_batch=True)
        print("profile")
        print("--- %s seconds ---" % (time.time() - start_time))

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
    
# Calculate----------------------------------------------------------------------------------------------------------------------
    
    def calculate_spectrogram(self, band, side, nperseg, noverlap, nfft, burst_size, subtract=None):
        if band == 'V':
            signal_type = 'complex'
        else:
            signal_type = 'real'

        burst = rpspy.get_band_signal(self.shot, self.file_path, band, side, signal_type, self.sweep - burst_size // 2, burst_size)
        f = func_aux.cached_get_linearization(self.shot, 24, band, shotfile_dir=self.file_path)
        f, linearized_burst = rpspy.linearize(f, burst)

        if band == 'V' and subtract == True:
            correction = func_aux.get_dispersion_phase()  # Eventually will be called with arguments like shot number, band, side, etc
            corrected_burst = linearized_burst - (2**11 + 1j * 2**11)  # Center around zero
            corrected_burst = corrected_burst * correction  # Multiply by dispersion phase
        else:
            corrected_burst = linearized_burst

        fs = rpspy.get_sampling_frequency(self.shot, self.file_path)

        f_beat, t, Sxx = spectrogram(
            corrected_burst, 
            fs=fs, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            nfft=nfft,
            return_onesided=False if corrected_burst.dtype == complex else True
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

# Parameters ---------------------------------------------------------------------------------------------------------------------

    def update_detector_params(self):
        sender = self.sender()

        self.params_filter.child('Low Filter').setValue(self.filters[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()][0], blockSignal=self.update_fft_params)
        self.params_filter.child('High Filter').setValue(self.filters[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()][1], blockSignal=self.update_fft_params)
        self.params_fft.child('nperseg').setValue(self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['nperseg'], blockSignal=self.update_fft_params)
        self.params_fft.child('noverlap').sigValueChanged.disconnect(self.update_fft_params)
        self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
        self.params_fft.child('noverlap').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('noverlap').setValue(self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['noverlap'], blockSignal=self.update_fft_params)
        self.params_fft.child('nfft').sigValueChanged.disconnect(self.update_fft_params)
        self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), np.inf))
        self.params_fft.child('nfft').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('nfft').setValue(self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['nfft'], blockSignal=self.update_fft_params)
        self.params_fft.child('Subtract dispersion').setValue(self.spect_params[self.params_detector.child('Side').value()][self.params_detector.child('Band').value()]['subtract'], blockSignal=self.update_fft_params)

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
            timestamp = rpspy.get_timestamps(self.shot, self.file_path)[value - 1]
            self.params_sweep.child('Sweep nº').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)

        elif sender == self.params_sweep.child('Sweep nº'):
            value = int(self.params_sweep.child('Sweep nº').value())
            timestamp = rpspy.get_timestamps(self.shot, self.file_path)[value - 1]
            self.params_sweep.child('Sweep nº').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)
            
        elif sender == self.params_sweep.child('Timestamp'):
            value = self.params_sweep.child('Timestamp').value()
            timestamp = func_aux.round_to_nearest(value, rpspy.get_timestamps(self.shot, self.file_path))
            index = np.where(rpspy.get_timestamps(self.shot, self.file_path) == timestamp)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep').setValue(index[0][0] + 1, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep nº').setValue(index[0][0] + 1, blockSignal=self.update_plot_params)
        
        self.update_plot()

        if not self.supress_updates:
            self.update_fft()
            self.update_all_beatf()
            self.update_profile()
        else:
            self.update_all_beatf()


    def update_fft_params(self):
        sender = self.sender()

        if sender == self.params_fft.child('nperseg'):
            value = int(self.params_fft.child('nperseg').value())
            self.spect_params[self.side][self.band]['nperseg'] = value
            self.params_fft.child('nperseg').setValue(value, blockSignal=self.update_fft_params)
            self.supress_updates = True
            self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
            self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), np.inf))
            self.supress_updates = False
        
        elif sender == self.params_fft.child('noverlap'):
            value = int(self.params_fft.child('noverlap').value())
            self.spect_params[self.side][self.band]['noverlap'] = value
            self.params_fft.child('noverlap').setValue(value, blockSignal=self.update_fft_params)
        
        elif sender == self.params_fft.child('nfft'):
            value = int(self.params_fft.child('nfft').value())
            self.spect_params[self.side][self.band]['nfft'] = value
            self.params_fft.child('nfft').setValue(value, blockSignal=self.update_fft_params)
        
        elif sender == self.params_fft.child('burst size (odd)'):
            value = int(self.params_fft.child('burst size (odd)').value())
            if value % 2 == 0:
                self.params_fft.child('burst size (odd)').setValue(value - 1, blockSignal=self.update_fft_params)
            else:
                self.params_fft.child('burst size (odd)').setValue(value, blockSignal=self.update_fft_params)

            lower_limit = int(1 + self.params_fft.child('burst size (odd)').value() // 2)
            upper_limit = int(len(rpspy.get_timestamps(self.shot, self.file_path)) - self.params_fft.child('burst size (odd)').value() // 2)
            self.supress_updates = True
            self.params_sweep.child('Sweep').setLimits([lower_limit, upper_limit])
            self.params_sweep.child('Sweep nº').setLimits([lower_limit, upper_limit])
            self.params_sweep.child('Timestamp').setLimits([rpspy.get_timestamps(self.shot, self.file_path)[lower_limit - 1], rpspy.get_timestamps(self.shot, self.file_path)[upper_limit - 1]])
            self.supress_updates = False

        elif sender == self.params_fft.child('Subtract dispersion'):
            self.spect_params[self.side][self.band]['subtract'] = self.params_fft.child('Subtract dispersion').value()

        elif sender == self.params_filter.child('Low Filter'):
            self.filters[self.side][self.band][0] = self.params_filter.child('Low Filter').value()
            if self.filters[self.side][self.band][0] + abs(self.f_beat[1] - self.f_beat[0]) >= self.filters[self.side][self.band][1]:
                self.filters[self.side][self.band][1] = self.filters[self.side][self.band][0] + abs(self.f_beat[1] - self.f_beat[0])
                self.params_filter.child('High Filter').setValue(self.filters[self.side][self.band][1], blockSignal=self.update_fft_params)
            self.draw_spectrogram()
        
        elif sender == self.params_filter.child('High Filter'):
            self.filters[self.side][self.band][1] = self.params_filter.child('High Filter').value()
            if self.filters[self.side][self.band][1] - abs(self.f_beat[1] - self.f_beat[0]) <= self.filters[self.side][self.band][0]:
                self.filters[self.side][self.band][0] = self.filters[self.side][self.band][1] - abs(self.f_beat[1] - self.f_beat[0])
                self.params_filter.child('Low Filter').setValue(self.filters[self.side][self.band][0], blockSignal=self.update_fft_params)
            self.draw_spectrogram()

        if not self.supress_updates:
            self.update_fft()
            self.update_one_beatf()
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