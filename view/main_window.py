from PyQt5.QtWidgets import QMainWindow, QHeaderView
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.parametertree import ParameterTree

from constants import WINDOW_SIZE, PARAMETER_TREE_WIDTH_PROPORTION, GRAPH_WIDTH_PROPORTION, DEFAULT_SECTION_SIZE


class MainWindowView(QMainWindow):
    """Main window layout. Creates docks, plots, parameter tree.

    Contains zero computation logic and zero signal wiring.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.setWindowIcon(QtGui.QIcon('reflecto-lab.png'))
        self.setWindowTitle('ReflectoLab')
        self.setGeometry(100, 100, WINDOW_SIZE[0], WINDOW_SIZE[1])

        # Dock area
        self.area = DockArea()
        self.setCentralWidget(self.area)

        # Create docks
        self.dock_tree = Dock("Settings", size=(WINDOW_SIZE[0] * PARAMETER_TREE_WIDTH_PROPORTION, WINDOW_SIZE[1]))
        self.dock_sweep = Dock(" ", size=(WINDOW_SIZE[0] * GRAPH_WIDTH_PROPORTION, WINDOW_SIZE[1] / 2))
        self.dock_spect = Dock(" ", size=(WINDOW_SIZE[0] * GRAPH_WIDTH_PROPORTION, WINDOW_SIZE[1] / 2))
        self.dock_beatf = Dock(" ", size=(WINDOW_SIZE[0] * GRAPH_WIDTH_PROPORTION, WINDOW_SIZE[1] / 2))
        self.dock_profile = Dock(" ", size=(WINDOW_SIZE[0] * GRAPH_WIDTH_PROPORTION, WINDOW_SIZE[1] / 2))

        # Add docks to the area
        self.area.addDock(self.dock_tree, 'left')
        self.area.addDock(self.dock_sweep, 'right', self.dock_tree)
        self.area.addDock(self.dock_spect, 'bottom', self.dock_sweep)
        self.area.addDock(self.dock_profile, 'right')
        self.area.addDock(self.dock_beatf, 'bottom', self.dock_profile)

        # Create plots
        self.plot_sweep = pg.PlotWidget(title="Sweep")
        self.dock_sweep.addWidget(self.plot_sweep)

        self.plot_spect = pg.PlotWidget(title="Spectrogram")
        self.dock_spect.addWidget(self.plot_spect)

        self.plot_beatf = pg.PlotWidget(title="Group Delay")
        self.dock_beatf.addWidget(self.plot_beatf)

        self.plot_profile = pg.PlotWidget(title="Profile")
        self.dock_profile.addWidget(self.plot_profile)

        # Parameter tree
        self.param_tree = ParameterTree()
        self.dock_tree.addWidget(self.param_tree)
        self.param_tree.header().setDefaultSectionSize(DEFAULT_SECTION_SIZE)
        self.param_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        # Color bar (lazily created by renderer)
        self.colorBar = None

    def set_parameter_panels(self, panels):
        """Add parameter groups to the tree."""
        for group in panels.all_groups():
            self.param_tree.addParameters(group)

    def show_post_load_params(self, panels):
        """Make parameter groups visible after shot load."""
        for group in panels.hideable_groups():
            group.setOpts(visible=True)

    def set_reconstruct_ui_enabled(self, panels, enabled):
        """Enable or disable reconstruction UI elements."""
        panels.reconstruct.child('Start Time').setOpts(enabled=enabled)
        panels.reconstruct.child('End Time').setOpts(enabled=enabled)
        panels.reconstruct.child('Time Step').setOpts(enabled=enabled)
        panels.reconstruct.child('Reconstruct Shot').setOpts(enabled=enabled)
