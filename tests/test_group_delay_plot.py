"""The group delay plot with bands that the shot does not have.

A band with no data has `f_probe = None`, and pyqtgraph's ``setData(None, None)`` raises a TypeError, so
the plot must draw nothing for it instead. pyqtgraph is replaced by a stand-in that behaves like that,
so no Qt is needed. Run from the reflecto-lab folder::

    python3 -m unittest discover -s tests -v
"""
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
RPSPY = os.environ.get("RPSPY_PATH", os.path.join(os.path.dirname(LAB), "rpspy"))
sys.path[:0] = [LAB, RPSPY, os.path.join(RPSPY, "tests")]

try:
    from synthetic_shot import _stub_missing_modules   # rpspy imports some AUG-only packages
except ImportError as e:  # pragma: no cover
    raise unittest.SkipTest(f"synthetic shot of rpspy not found in {RPSPY}: {e}")
_stub_missing_modules()


class Curve:
    """What the plot widget returns from plot(): setData takes two lists or arrays, like pyqtgraph's."""

    def __init__(self):
        self.x = self.y = None

    def setData(self, *args):
        if len(args) == 2 and not all(isinstance(a, (list, np.ndarray)) for a in args):
            raise TypeError('When passing two unnamed arguments, both must be a list or array of values. '
                            f'(got {type(args[0])}, {type(args[1])})')
        self.x, self.y = (np.asarray(a) for a in args)

    def setZValue(self, z):
        pass


class Widget:
    def __init__(self):
        self.curves = []

    def plot(self, **kwargs):
        self.curves.append(Curve())
        return self.curves[-1]

    def setLabel(self, *args, **kwargs):
        pass


def fake_pyqtgraph():
    pg = types.ModuleType('pyqtgraph')
    pg.mkPen = lambda **kwargs: kwargs
    qt = types.ModuleType('pyqtgraph.Qt')
    qt.QtCore = qt.QtGui = qt.QtWidgets = mock.MagicMock()
    pg.Qt = qt
    return {'pyqtgraph': pg, 'pyqtgraph.Qt': qt}


class GroupDelayPlotTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.modules.pop('view.plot_renderers', None)
        cls._patch = mock.patch.dict(sys.modules, fake_pyqtgraph())
        cls._patch.start()
        from view.plot_renderers import PlotRenderer
        from model.state import BeatFrequencyData, AggregatedDelayData, FrequencyExclusion
        cls.PlotRenderer, cls.Beat, cls.Agg, cls.Excl = PlotRenderer, BeatFrequencyData, AggregatedDelayData, FrequencyExclusion
        from constants import BANDS, SIDES
        cls.BANDS, cls.SIDES = BANDS, SIDES

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop('view.plot_renderers', None)
        cls._patch.stop()

    def beat_frequencies(self, have):
        f = np.linspace(20e9, 30e9, 50)
        return {side: {band: self.Beat(f_probe=f, y_beatf=np.ones(50), y_beat_time=np.ones(50) * 1e-9)
                       if (band, side) in have else self.Beat()
                       for band in self.BANDS} for side in self.SIDES}

    def draw(self, have, exclusions):
        widget, renderer = Widget(), self.PlotRenderer()
        agg = self.Agg(f_probe=np.arange(3.), beat_time=np.arange(3.))
        renderer.draw_group_delays(widget, self.beat_frequencies(have), exclusions, agg, agg, 0.005)
        return renderer

    def test_bands_without_data_draw_nothing(self):
        have = {('K', 'HFS'), ('Q', 'LFS'), ('V', 'LFS')}
        renderer = self.draw(have, {side: [] for side in self.SIDES})
        for side in self.SIDES:
            for band in self.BANDS:
                curve = renderer._gd_band[(side, band)]
                if (band, side) in have:
                    self.assertEqual(len(curve.x), 50)
                else:
                    self.assertEqual(len(curve.x), 0)

    def test_bands_without_data_draw_nothing_with_an_active_exclusion(self):
        have = {('K', 'HFS'), ('W', 'LFS')}
        exclusions = {'HFS': [self.Excl(22e9, 24e9, 0.0, 10.0, True)], 'LFS': []}
        renderer = self.draw(have, exclusions)
        for band in self.BANDS:
            overlay = renderer._gd_excl[('HFS', band)][0]
            self.assertEqual(len(overlay.x) > 0, band == 'K')

    def test_the_stand_in_rejects_none_like_pyqtgraph(self):
        with self.assertRaises(TypeError):
            Curve().setData(None, None)


if __name__ == '__main__':
    unittest.main()
