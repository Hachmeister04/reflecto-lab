import sys
import numpy as np
from scipy.signal import spectrogram
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QSplitter
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
from rpspy import get_band_signal, get_timestamps, get_sampling_frequency, column_wise_max_with_quadratic_interpolation
from func_aux import round_to_nearest, get_shot_from_path, find_path_from_shot
import time

#TODO: Remove hardcoded values and add them here
MAX_BURST_SIZE = 285
DEFAULT_NPERSEG = 256
DEFAULT_NOVERLAP = 220
DEFAULT_NFFT = 512
MIN_NPERSEG = 10
MAX_NFFT = np.inf

#TODO: Make all calls to pyqtgraph directly. Do not use Qt to draw windows
class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        #---------------------------------------------------------------------------

        # Set up the main window
        self.setWindowTitle('PyQtGraph Plot with Slider')
        self.setGeometry(100, 100, 1600, 800)
        
        # Create layouts and widgets------------------------------------------------

        # Create a central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        # Create a QSplitter for adjustable layout
        self.splitter = QSplitter(Qt.Horizontal, self.central_widget)

        # Create the main graphics layout widget
        self.graph_layout = pg.GraphicsLayoutWidget()
        
        #TODO: Fix the displayed value in the parameters
        #TODO: Add the parameters to a different file
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
            {'name': 'Sweep nº', 'type': 'float', 'value': 1},
            {'name': 'Timestamp', 'type': 'float', 'value': 0, 'suffix': 's', 'siPrefix': True},
        ])
        self.params_fft = Parameter.create(name='FFT', type='group', children=[
            {'name': 'nperseg', 'type': 'float', 'value': DEFAULT_NPERSEG},
            {'name': 'noverlap', 'type': 'float', 'value': DEFAULT_NOVERLAP},
            {'name': 'nfft', 'type': 'float', 'value': DEFAULT_NFFT},
            {'name': 'burst size (odd)', 'type': 'float', 'value': 1, 'limits': (1, MAX_BURST_SIZE)},
            {'name': 'cmap', 'type': 'cmaplut', 'value': 'plasma'}
        ])
        self.param_tree = ParameterTree()
        self.param_tree.addParameters(self.params_file)

        # Create the first plot
        self.plot_sweep = self.graph_layout.addPlot(title="Sweep")
        
        # Create the second plot next to the first one
        self.plot_fft = self.graph_layout.addPlot(title="FFT")
        #self.plot_fft.setFixedWidth(600)

        # Add widgets and layouts---------------------------------------------------
        
        # Add widgets to the splitter
        self.splitter.addWidget(self.param_tree)
        self.splitter.addWidget(self.graph_layout)
        
        # Add the splitter to the main layout
        self.layout.addWidget(self.splitter)

        #---------------------------------------------------------------------------

        # Connect the file to the function that displays the rest of the parameters and the graphs
        self.params_file.child('Open').sigValueChanged.connect(self.update_display)
        self.params_file.child('Shot').sigValueChanged.connect(self.update_display)
    
    def update_display(self):
        sender = self.sender()

        # Name the path to the directory
        if sender == self.params_file.child('Open'):
            self.file_path = self.params_file.child('Open').value()
            self.shot = get_shot_from_path(self.file_path)
        elif sender == self.params_file.child('Shot'):
            self.file_path = find_path_from_shot(self.params_file.child('Shot').value())
            self.shot = self.params_file.child('Shot').value()
        
        try:
            if self.params_added == False:
                self.param_tree.addParameters(self.params_detector)
                self.param_tree.addParameters(self.params_sweep)
                self.param_tree.addParameters(self.params_fft)
        except AttributeError:
            self.param_tree.addParameters(self.params_detector)
            self.param_tree.addParameters(self.params_sweep)
            self.param_tree.addParameters(self.params_fft)
            self.params_added = True

        # Connect the parameters to the functions-----------------------------------

        # Connect the lists to update the plot
        self.params_detector.child('Band').sigValueChanged.connect(self.update_plot)
        self.params_detector.child('Side').sigValueChanged.connect(self.update_plot)

        # Connect the slider, sweep, and timestamp to update the plot
        self.params_sweep.child('Sweep').sigValueChanged.connect(self.update_plot_params)
        self.params_sweep.child('Sweep nº').sigValueChanged.connect(self.update_plot_params)
        self.params_sweep.child('Timestamp').sigValueChanged.connect(self.update_plot_params)

        #Connect the fft params to update the fft
        self.params_fft.child('nperseg').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('noverlap').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('nfft').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('burst size (odd)').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('cmap').sigValueChanged.connect(self.update_fft_params)

        #---------------------------------------------------------------------------

        self.update_plot()

        # Set limits to the parameters----------------------------------------------

        self.params_sweep.child('Sweep').setLimits((1, len(get_timestamps(self.shot, self.file_path))))
        self.params_sweep.child('Sweep nº').setLimits((1, len(get_timestamps(self.shot, self.file_path))))
        self.params_fft.child('nperseg').setLimits((MIN_NPERSEG, len(self.data)))
        self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
        self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), MAX_NFFT))
        
    def update_plot(self):
        self.band = self.params_detector.child('Band').value()
        self.side = self.params_detector.child('Side').value()
        if self.band == 'V':
            self.signal = 'complex'
        else:
            self.signal = 'real'

        self.sweep = int(self.params_sweep.child('Sweep nº').value()) - 1
        self.data = get_band_signal(self.shot, self.file_path, self.band, self.side, self.signal, self.sweep)[0]
        x = np.arange(len(self.data)) / get_sampling_frequency(self.shot, self.file_path)
        y = np.real(self.data)
        # Plot the data
        self.plot_sweep.clear()  # Clears the plot
        self.plot_sweep.plot(x, y, pen=pg.mkPen(color='r', width=2))
        self.plot_sweep.setLabel('bottom', 'Time', units='s')
        print("plot")
        #TODO: Don't call update_fft inside this function
        self.update_fft()
    
    def update_fft(self):
        start_time = time.time()

        # Compute the spectrogram of the data
        burst_size = int(self.params_fft.child('burst size (odd)').value())
        burst = get_band_signal(self.shot, self.file_path, self.band, self.side, self.signal, self.sweep - burst_size // 2, burst_size)
        
        nperseg = int(self.params_fft.child('nperseg').value())
        noverlap = int(self.params_fft.child('noverlap').value())
        nfft = int(self.params_fft.child('nfft').value())
        colormap = self.params_fft.child('cmap').value()

        fs = get_sampling_frequency(self.shot, self.file_path)  # Sampling frequency
        f, t, Sxx = spectrogram(np.real(burst), fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
        # Calculate average of the burst
        Sxx = np.average(Sxx, axis=0)

        # Example: Transformed display of ImageItem
        alpha_x = (nperseg-noverlap)/fs
        tr = QtGui.QTransform() # prepare ImageItem transformation
        tr.translate(noverlap/2/fs, 0)
        tr.scale(alpha_x, f[1]) # scale horizontal and vertical axes

        i1 = pg.ImageItem(image=Sxx.T) # Note: `Sxx` needs to be transposed to fit the display format
        i1.setTransform(tr) # assign transform
        
        self.plot_fft.clear()  # Clear previous plot
        self.plot_fft.addItem(i1)
        
        # Set up color bar
        try:
            self.colorBar.setImageItem(i1)
            self.colorBar.setColorMap(colormap)
        except AttributeError:
            self.colorBar = self.plot_fft.addColorBar(i1, colorMap=colormap, values=(0, np.max(Sxx)))

        # Generate the line through the max of the graph
        x = np.linspace(t[0], t[-1], len(t))  # X coordinates
        y, _ = column_wise_max_with_quadratic_interpolation(Sxx)  # Y coordinates
        y *= f[1]
        self.plot_fft.plot(x, y, pen=pg.mkPen(color='r', width=1))
        
        # Configure plot appearance
        self.plot_fft.setMouseEnabled(x=True, y=True)
        self.plot_fft.disableAutoRange()
        #self.plot_fft.hideButtons()
        #self.plot_fft.setRange(xRange=(0, 25e-6), yRange=(0, 2.5e6), padding=0)
        self.plot_fft.showAxes(True, showValues=(True, False, False, True))
        self.plot_fft.setLabel('bottom', 'Time', units='s')
        self.plot_fft.setLabel('left', 'Frequency', units='Hz')
        print("fft")
        print("--- %s seconds ---" % (time.time() - start_time))

    def update_plot_params(self):
        #TODO: Don't redraw the graphs if the values are the same
        sender = self.sender()

        if sender == self.params_sweep.child('Sweep'):
            value = self.params_sweep.child('Sweep').value()
            timestamp = get_timestamps(self.shot, self.file_path)[value - 1]
            self.params_sweep.child('Sweep nº').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)

        elif sender == self.params_sweep.child('Sweep nº'):
            value = int(self.params_sweep.child('Sweep nº').value())
            timestamp = get_timestamps(self.shot, self.file_path)[value - 1]
            self.params_sweep.child('Sweep nº').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep').setValue(value, blockSignal=self.update_plot_params)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)
            
        elif sender == self.params_sweep.child('Timestamp'):
            value = self.params_sweep.child('Timestamp').value()
            timestamp = round_to_nearest(value, get_timestamps(self.shot, self.file_path))
            index = np.where(get_timestamps(self.shot, self.file_path) == timestamp)
            self.params_sweep.child('Timestamp').setValue(timestamp, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep').setValue(index[0][0] + 1, blockSignal=self.update_plot_params)
            self.params_sweep.child('Sweep nº').setValue(index[0][0] + 1, blockSignal=self.update_plot_params)
        
        self.update_plot()

    #TODO: Optimize this function so that it doesn't fft the same thing more than once
    def update_fft_params(self):
        sender = self.sender()

        if sender == self.params_fft.child('burst size (odd)'):
            value = int(self.params_fft.child('burst size (odd)').value())
            if value % 2 == 0:
                self.params_fft.child('burst size (odd)').setValue(value - 1, blockSignal=self.update_fft_params)
            else:
                self.params_fft.child('burst size (odd)').setValue(value, blockSignal=self.update_fft_params)
            lower_limit = int(1 + self.params_fft.child('burst size (odd)').value() // 2)
            upper_limit = int(len(get_timestamps(self.shot, self.file_path)) - self.params_fft.child('burst size (odd)').value() // 2)
            self.params_sweep.child('Sweep').setLimits([lower_limit, upper_limit])
            self.params_sweep.child('Sweep nº').setLimits([lower_limit, upper_limit])
            self.params_sweep.child('Timestamp').setLimits([get_timestamps(self.shot, self.file_path)[lower_limit - 1], get_timestamps(self.shot, self.file_path)[upper_limit - 1]])

        elif sender == self.params_fft.child('nperseg'):
            value = int(self.params_fft.child('nperseg').value())
            self.params_fft.child('nperseg').setValue(value, blockSignal=self.update_fft_params)
            self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
            self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), np.inf))
        
        elif sender == self.params_fft.child('noverlap'):
            value = int(self.params_fft.child('noverlap').value())
            self.params_fft.child('noverlap').setValue(value, blockSignal=self.update_fft_params)
        
        elif sender == self.params_fft.child('nfft'):
            value = int(self.params_fft.child('nfft').value())
            self.params_fft.child('nfft').setValue(value, blockSignal=self.update_fft_params)
        
        self.update_fft()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = PlotWindow()
    main_window.show()
    sys.exit(app.exec_())