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

""" import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox

class SimpleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dropdown Example")
        self.setGeometry(100, 100, 400, 300)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a combo box (dropdown list)
        self.combo = QComboBox()
        self.combo.addItems(["Option 1", "Option 2", "Option 3"])

        # Add combo box to layout
        layout.addWidget(self.combo)

        # Connect the combo box selection change
        self.combo.currentTextChanged.connect(self.on_selection_change)

    def on_selection_change(self, text):
        print(f"Selected: {text}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleWindow()
    window.show()
    sys.exit(app.exec_())

 """


"""
This example demonstrates the use of pyqtgraph's parametertree system. This provides
a simple way to generate user interfaces that control sets of parameters. The example
demonstrates a variety of different parameter types (int, float, list, etc.)
as well as some customized parameter types
"""

