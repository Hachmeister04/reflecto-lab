""" from rpspy import get_band_signal, get_timestamps


print(get_band_signal(40112, "C:/Users/filip/Desktop/PyQTgraph/40112", "K", "lfs", "real", 0))

print(get_timestamps(40112, "C:/Users/filip/Desktop/PyQTgraph/40112")) """

import pyqtgraph.examples
pyqtgraph.examples.run()

""" import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create the main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # Create the parameter tree with one int parameter
        params = [
            {'name': 'Integer Parameter', 'type': 'int', 'value': 10}
        ]
        parameter = Parameter.create(name='params', type='group', children=params)
        
        # Create the parameter tree widget
        param_tree = ParameterTree()
        param_tree.setParameters(parameter, showTop=True)
        
        # Add the parameter tree to the layout
        layout.addWidget(param_tree)
        
        # Set the central widget
        self.setCentralWidget(main_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())




import pyqtgraph
import numpy

print(pyqtgraph.Qt.VERSION_INFO)
print(numpy.__version__) """