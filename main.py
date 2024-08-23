import sys
import os
import numpy as np
from scipy.signal import spectrogram
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QHBoxLayout, QSplitter
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
from rpspy import get_band_signal, get_timestamps, get_sampling_frequency, column_wise_max_with_quadratic_interpolation
from func_aux import round_to_nearest

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        #---------------------------------------------------------------------------

        # Name the path to the directory
        self.shot = 40112
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.file_path = self.current_directory + "/../" + str(self.shot)

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

        # Create the main graphics and slider widget and a layout for it
        self.graph_slider_widget = QWidget()
        self.graph_slider_layout = QVBoxLayout(self.graph_slider_widget)

        # Create the main graphics layout widget
        self.graph_layout = pg.GraphicsLayoutWidget()
        
        # Create the parameter tree
        self.params_detector = Parameter.create(name='Detector', type='group', children=[
            {'name': 'Band', 'type': 'list', 'limits': ['K', 'Ka', 'Q', 'V']},
            {'name': 'Side', 'type': 'list', 'limits': ['hfs', 'lfs']}
        ])
        self.params_sweep = Parameter.create(name='Sweep', type='group', children=[
            {'name': 'Sweep nº', 'type': 'float', 'value': 1, 'limits': (1, len(get_timestamps(self.shot, self.file_path)))},
            {'name': 'Timestamp', 'type': 'float', 'value': 0, 'suffix': 's', 'siPrefix': True},
        ])
        self.params_fft = Parameter.create(name='FFT', type='group', children=[
            {'name': 'nperseg', 'type': 'int', 'value': 256},
            {'name': 'noverlap', 'type': 'int', 'value': 255},
            {'name': 'nfft', 'type': 'int', 'value': 2048},
            {'name': 'cmap', 'type': 'cmaplut', 'value': 'plasma'}
        ])
        self.param_tree = ParameterTree()
        self.param_tree.addParameters(self.params_detector, showTop=True)
        self.param_tree.addParameters(self.params_sweep, showTop=True)
        self.param_tree.addParameters(self.params_fft, showTop=True)

        # Create the first plot
        self.plot_sweep = self.graph_layout.addPlot(title="Sweep")
        
        # Create the second plot next to the first one
        self.plot_fft = self.graph_layout.addPlot(title="FFT")
        #self.plot_fft.setFixedWidth(600)
        
        # Create and add the slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(len(get_timestamps(self.shot, self.file_path)))
        self.slider.setValue(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.graph_slider_layout.addWidget(self.slider)
        
        # Add widgets and layouts---------------------------------------------------

        # Add the graph and the slider to the graph and slider layout
        self.graph_slider_layout.addWidget(self.graph_layout)
        self.graph_slider_layout.addWidget(self.slider)
        
        # Add widgets to the splitter
        self.splitter.addWidget(self.param_tree)
        self.splitter.addWidget(self.graph_slider_widget)
        
        # Add the splitter to the main layout
        self.layout.addWidget(self.splitter)

        #---------------------------------------------------------------------------

        self.update_plot()

        #---------------------------------------------------------------------------

        # Connect the lists to update the plot
        self.params_detector.child('Band').sigValueChanged.connect(self.update_plot)
        self.params_detector.child('Side').sigValueChanged.connect(self.update_plot)

        # Connect the slider, sweep, and timestamp to update the plot
        self.slider.valueChanged.connect(self.update_plot_params)
        self.params_sweep.child('Sweep nº').sigValueChanged.connect(self.update_plot_params)
        self.params_sweep.child('Timestamp').sigValueChanged.connect(self.update_plot_params)

        #Connect the fft params to update the fft
        self.params_fft.child('nperseg').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('noverlap').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('nfft').sigValueChanged.connect(self.update_fft_params)
        self.params_fft.child('cmap').sigValueChanged.connect(self.update_fft)
    
    def update_plot(self):
        # Update the plot based on the slider value
        band = self.params_detector.child('Band').value()
        side = self.params_detector.child('Side').value()
        if band == 'V':
            signal = 'complex'
        else:
            signal = 'real'

        value = self.slider.value()
        sweep = value - 1
        self.data = get_band_signal(40112, self.file_path, band, side, signal, sweep)[0]
        x = np.arange(len(self.data)) / get_sampling_frequency(self.shot, self.file_path)
        y = np.real(self.data)
        # Plot the data
        self.plot_sweep.clear()  # Clears the plot
        self.plot_sweep.plot(x, y, pen=pg.mkPen(color='r', width=2))
        self.plot_sweep.setLabel('bottom', 'Time', units='s')
        self.update_fft()
    
    def update_fft(self):
        # Compute the spectrogram of the data
        nperseg = self.params_fft.child('nperseg').value()
        noverlap = self.params_fft.child('noverlap').value()
        nfft = self.params_fft.child('nfft').value()
        colormap = self.params_fft.child('cmap').value()

        fs = get_sampling_frequency(self.shot, self.file_path)  # Sampling frequency
        f, t, Sxx = spectrogram(np.real(self.data), fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
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


    #TODO: Optimize this function so that it doesn't plot the same thing more than once
    def update_plot_params(self):
        sender = self.sender()

        if sender == self.slider:
            print("a")
            value = self.slider.value()
            timestamp = get_timestamps(self.shot, self.file_path)[value - 1]
            self.params_sweep.child('Sweep nº').setValue(value)
            self.params_sweep.child('Timestamp').setValue(timestamp)

        elif sender == self.params_sweep.child('Sweep nº'):
            print("b")
            value = int(self.params_sweep.child('Sweep nº').value())
            self.params_sweep.child('Sweep nº').setValue(value)
            self.slider.setValue(value)
            timestamp = get_timestamps(self.shot, self.file_path)[value - 1]
            self.params_sweep.child('Timestamp').setValue(timestamp)
            
        elif sender == self.params_sweep.child('Timestamp'):
            print("c")
            value = self.params_sweep.child('Timestamp').value()
            timestamp = round_to_nearest(value, get_timestamps(self.shot, self.file_path))
            index = np.where(get_timestamps(self.shot, self.file_path) == timestamp)
            self.slider.setValue(index[0][0] + 1)
            self.params_sweep.child('Sweep nº').setValue(index[0][0] + 1)
        
        print("plot")
        self.update_plot()

    def update_fft_params(self):
        self.params_fft.child('nperseg').setLimits((10, len(self.data)))
        self.params_fft.child('noverlap').setLimits((0, self.params_fft.child('nperseg').value() - 1))
        self.params_fft.child('nfft').setLimits((self.params_fft.child('nperseg').value(), np.inf))
        

        if self.params_fft.child('noverlap').value() < self.params_fft.child('nperseg').value(): #This condition because the setdefault of the noverlap has some strange order of events
            print("fft")
            self.update_fft()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = PlotWindow()
    main_window.show()
    sys.exit(app.exec_())
