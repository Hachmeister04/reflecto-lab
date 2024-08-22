import sys
import os
import numpy as np
from scipy.signal import spectrogram
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QHBoxLayout, QSplitter
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
from rpspy import get_band_signal, get_timestamps, get_sampling_frequency
from func_aux import round_to_nearest

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        #---------------------------------------------------------------------------

        # Name the path to the directory
        self.shot = 40112
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.file_path = self.current_directory + "/" + str(self.shot)

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
        self.params_sweep = Parameter.create(name='Sweep', type='group', children=[
            {'name': 'Sweep nº', 'type': 'float', 'value': 1, 'limits': (1, len(get_timestamps(self.shot, self.file_path)))},
            {'name': 'Timestamp', 'type': 'float', 'value': 0},
        ])
        #TODO: Correct valueErrors when typing the wrong input in these parameters
        self.params_fft = Parameter.create(name='FFT', type='group', children=[
            {'name': 'nperseg', 'type': 'int', 'value': 256},
            {'name': 'noverlap', 'type': 'int', 'value': 255},
            {'name': 'nfft', 'type': 'int', 'value': 2048}
        ])
        self.param_tree = ParameterTree()
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

        # Connect the slider to update the plot
        self.slider.valueChanged.connect(self.update_from_slider)

        # Connect the parameter tree to update the plot
        self.params_sweep.child('Sweep nº').sigValueChanged.connect(self.update_from_index)
        self.params_sweep.child('Timestamp').sigValueChanged.connect(self.update_from_timestamp)

        #Connect the parameter tree to update the fft
        self.params_fft.child('nperseg').sigValueChanged.connect(self.update_fft)
        self.params_fft.child('noverlap').sigValueChanged.connect(self.update_fft)
        self.params_fft.child('nfft').sigValueChanged.connect(self.update_fft)
    
    def update_plot(self):
        # Update the plot based on the slider value
        value = self.slider.value()
        self.plot_sweep.clear()  # Clears the plot
        self.sweep = value - 1
        self.data = get_band_signal(40112, self.file_path, "K", "lfs", "real", self.sweep)
        self.x = np.arange(len(self.data[0])) / get_sampling_frequency(self.shot, self.file_path)
        self.y = self.data[0]
        # Plot the data
        self.plot_sweep.plot(self.x, self.y, pen=pg.mkPen(color='r', width=2))
        self.plot_sweep.setLabel('bottom', 'Time', units='s')
        self.update_fft()
    
    def update_fft(self):
        #TODO: Find out where the x-axis starts (translation to the right)

        # Compute the spectrogram of the data
        self.nperseg = self.params_fft.child('nperseg').value()
        self.noverlap = self.params_fft.child('noverlap').value()
        self.nfft = self.params_fft.child('nfft').value()

        fs = get_sampling_frequency(self.shot, self.file_path)  # Sampling frequency
        f, t, Sxx = spectrogram(self.y, fs=fs, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
        
        # Note: `Sxx` needs to be transposed to fit the display format
        
        # Example: Transformed display of ImageItem
        tr = QtGui.QTransform() # prepare ImageItem transformation
        tr.scale(t[1] - t[0], f[1]) # scale horizontal and vertical axes
        tr.translate(len(t)/(t[-1]-t[0])*1/fs*self.noverlap/2, 0)

        i1 = pg.ImageItem(image=Sxx.T)
        i1.setTransform(tr) # assign transform

        self.plot_fft.clear()  # Clear previous plot
        self.plot_fft.addItem(i1)
        
        # Set up color bar
        try:
            self.colorBar.setImageItem(i1)
        except AttributeError:
            self.colorBar = self.plot_fft.addColorBar(i1, colorMap='CET-L9', values=(0, np.max(Sxx)))

        # Generate random dots and connect them with a line
        num_points = 10
        random_x = np.linspace(t[0], t[-1], num_points)  # X coordinates spaced over the time axis
        random_y = np.random.uniform(low=0, high=5e6, size=num_points)  # Y coordinates are random values

        self.plot_fft.plot(random_x, random_y, pen=pg.mkPen(color='g', width=2), symbol='o', symbolSize=8, symbolBrush=('r'))
        
        # Configure plot appearance
        self.plot_fft.setMouseEnabled(x=True, y=True)
        self.plot_fft.disableAutoRange()
        self.plot_fft.hideButtons()
        #self.plot_fft.setRange(yRange=(0, 2.5e6), padding=0)
        self.plot_fft.showAxes(True, showValues=(True, False, False, True))
        self.plot_fft.setLabel('bottom', 'Time', units='s')
        self.plot_fft.setLabel('left', 'Frequency', units='Hz')

    #TODO: Find a way to connect the 3 components (slider, index, and timestamp) another way
    def update_from_slider(self):
        # Update the line edits based on the slider value
        value = self.slider.value()
        timestamp = get_timestamps(self.shot, self.file_path)[value - 1]
        self.params_sweep.child('Sweep nº').setValue(value)
        self.params_sweep.child('Timestamp').setValue(timestamp)
        self.update_plot()

    def update_from_index(self):
        try:
            value = int(self.params_sweep.child('Sweep nº').value())
            self.params_sweep.child('Sweep nº').setValue(value)
            self.slider.setValue(value)
            
        except ValueError:
            pass  # Ignore invalid input
    
    def update_from_timestamp(self):
        try:
            value = self.params_sweep.child('Timestamp').value()
            timestamp = round_to_nearest(value, get_timestamps(self.shot, self.file_path))
            index = np.where(get_timestamps(self.shot, self.file_path) == timestamp)
            self.slider.setValue(index[0][0] + 1)
        except ValueError:
            pass  # Ignore invalid input

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = PlotWindow()
    main_window.show()
    sys.exit(app.exec_())
