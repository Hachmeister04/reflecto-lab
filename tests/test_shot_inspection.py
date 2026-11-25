"""What the user is told about a shot before it is loaded, and the choice of bands that follows.

The shot is inspected (raw signals, entries of each band in the linearization tables), the user chooses which
bands to process, and the shot is loaded with those. No Qt is needed: the dialog only shows what the inspection says.
Run from the reflecto-lab folder::

    python3 -m unittest discover -s tests -v
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
RPSPY = os.environ.get("RPSPY_PATH", os.path.join(os.path.dirname(LAB), "rpspy"))
sys.path[:0] = [LAB, RPSPY, os.path.join(RPSPY, "tests")]

try:
    from synthetic_shot import make_synthetic_shot, install_environment
except ImportError as e:  # pragma: no cover
    raise unittest.SkipTest(f"synthetic shot of rpspy not found in {RPSPY}: {e}")


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class ShotInspectionTest(unittest.TestCase):
    """Shot A has W recorded but no table entries for it. Shot B has tables for every band."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.a = make_synthetic_shot(cls._tmp.name, shot=45060, n_sweeps=200, with_w=True, no_table_bands=('W',), seed=0)
        cls.b = make_synthetic_shot(cls._tmp.name, shot=45061, n_sweeps=200, with_w=True, seed=1)
        cls.c = make_synthetic_shot(cls._tmp.name, shot=45062, n_sweeps=200, seed=2)    # no W at all
        install_environment(cls.a)
        import rpspy
        from model.shot_model import ShotModel
        from model import shot_inspection as si
        cls.rpspy, cls.ShotModel, cls.si = rpspy, ShotModel, si

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def inspect(self, shot, linearization_shot=None):
        return self.ShotModel().inspect_shot(shot.shot, shot.shot_dir, linearization_shot)

    # --- the inspection -----------------------------------------------------------------------

    def test_a_band_without_entries_is_flagged(self):
        insp = self.inspect(self.a)
        self.assertEqual(insp.entries('W'), 0)
        self.assertGreater(insp.entries('K'), 0)
        self.assertEqual(insp.state('W', 'LFS'), self.si.NO_TABLE)
        self.assertEqual(insp.state('K', 'HFS'), self.si.USABLE)
        self.assertEqual(insp.state('W', 'HFS'), self.si.NO_SIGNAL)          # W has no HFS signal
        self.assertEqual(insp.bands_without_table(), ['W'])

    def test_bands_that_cannot_be_processed_are_not_selected_by_default(self):
        insp = self.inspect(self.a)
        selection = insp.default_selection()
        self.assertNotIn(('W', 'LFS'), selection)
        self.assertEqual(selection, {(b.name, b.side) for b in insp.raw_bands if b.name != 'W'})
        self.assertEqual(len(selection), 8)

    def test_a_shot_without_w_signal_has_nothing_to_flag(self):
        insp = self.inspect(self.c)
        self.assertEqual(insp.bands_without_table(), [])
        self.assertEqual(insp.state('W', 'LFS'), self.si.NO_SIGNAL)
        self.assertEqual(insp.entries('W'), 0)

    def test_the_tables_of_another_shot(self):
        insp = self.inspect(self.a, self.b.shot)
        self.assertEqual(insp.linearization_shot, self.b.shot)
        self.assertGreater(insp.entries('W'), 0)
        self.assertEqual(insp.state('W', 'LFS'), self.si.USABLE)
        self.assertIn(('W', 'LFS'), insp.default_selection())
        self.assertEqual(insp.bands_without_table(), [])

    def test_rows_have_one_entry_per_band_of_the_diagnostic(self):
        rows = self.inspect(self.a).rows()
        self.assertEqual([name for name, _, _ in rows], ['K', 'Ka', 'Q', 'V', 'W'])
        w = dict((name, (states, entries)) for name, states, entries in rows)['W']
        self.assertEqual(w, ({'HFS': self.si.NO_SIGNAL, 'LFS': self.si.NO_TABLE}, 0))

    def test_the_report_says_what_the_user_needs_to_know(self):
        text = self.inspect(self.a).report()
        self.assertIn("Shot 45060", text)
        self.assertIn("K-HFS", text)
        self.assertIn("W-LFS", text)                       # the signal is there
        self.assertIn("W band - no entries available", text)
        self.assertRegex(text, r"K band - \d+ entries")
        self.assertNotIn("instead of", text)

    def test_the_report_of_a_borrowed_table(self):
        text = self.inspect(self.a, self.b.shot).report()
        self.assertIn("Linearization tables of shot 45061", text)
        self.assertIn("instead of those of shot 45060", text)
        self.assertNotIn("no entries available", text)

    def test_a_shot_without_tables_is_reported_not_raised(self):
        insp = self.inspect(self.a, 99)
        self.assertIsNotNone(insp.table_error)
        self.assertIn("shot 99", insp.table_error)
        self.assertEqual(insp.default_selection(), set())
        self.assertIn("Linearization tables of shot 99", insp.report())
        self.assertIn("no linearization tables", insp.report().lower().replace("there are ", ""))

    def test_inspecting_does_not_change_the_model(self):
        m = self.ShotModel()
        m.inspect_shot(self.a.shot, self.a.shot_dir)
        self.assertEqual((m.shot, m.used_bands, m.raw_bands, m.linearization_reference_shot), (0, [], [], 0))

    def test_a_shot_without_raw_signals_cannot_be_inspected(self):
        with tempfile.TemporaryDirectory() as root:
            empty = make_synthetic_shot(root, shot=45063, n_sweeps=8)
            for name in os.listdir(empty.shot_dir):
                if name.endswith('.sig'):
                    os.remove(os.path.join(empty.shot_dir, name))
            with self.assertRaises(FileNotFoundError):
                self.ShotModel().inspect_shot(empty.shot, empty.shot_dir)
        install_environment(self.a)

    # --- loading with a choice of bands ---------------------------------------------------------

    def load(self, shot, selected=None, linearization_shot=None):
        m = self.ShotModel()
        with quiet():
            m.load_shot(m.inspect_shot(shot.shot, shot.shot_dir, linearization_shot), selected)
        return m

    def test_by_default_everything_that_can_be_processed_is_loaded(self):
        m = self.load(self.a)
        self.assertEqual(len(m.used_bands), 8)
        self.assertNotIn('W', m.used_band_names)
        self.assertEqual(m.bands_on_side('LFS'), ['K', 'Ka', 'Q', 'V'])
        self.assertEqual(len(m.raw_bands), 9)                               # W is still known to be recorded
        self.assertEqual(m.linearization_reference_shot, self.a.shot)

    def test_bands_can_be_left_out(self):
        selected = {('K', 'HFS'), ('K', 'LFS'), ('Q', 'LFS')}
        m = self.load(self.a, selected)
        self.assertEqual({(b.name, b.side) for b in m.used_bands}, selected)
        self.assertEqual(m.used_sides, ['HFS', 'LFS'])
        self.assertEqual(m.bands_on_side('HFS'), ['K'])
        self.assertEqual(m.bands_on_side('LFS'), ['K', 'Q'])

    def test_a_band_without_entries_cannot_be_forced_in(self):
        m = self.load(self.a, {('K', 'HFS'), ('W', 'LFS')})
        self.assertEqual([(b.name, b.side) for b in m.used_bands], [('K', 'HFS')])

    def test_selecting_nothing_is_an_error_and_changes_nothing(self):
        m = self.load(self.c)
        before = (m.shot, list(m.used_bands), m.linearization_reference_shot)
        with self.assertRaisesRegex(ValueError, "No band is selected"):
            m.load_shot(m.inspect_shot(self.a.shot, self.a.shot_dir), set())
        self.assertEqual((m.shot, list(m.used_bands), m.linearization_reference_shot), before)

    def test_a_shot_where_no_band_can_be_processed_is_an_error(self):
        m = self.ShotModel()
        with self.assertRaisesRegex(ValueError, "No band of shot 45060 can be processed"):
            m.load_shot(m.inspect_shot(self.a.shot, self.a.shot_dir, 99))
        self.assertEqual(m.shot, 0)

    def test_the_tables_of_another_shot_make_the_band_usable(self):
        m = self.load(self.a, linearization_shot=self.b.shot)
        self.assertIn('W', m.used_band_names)
        self.assertEqual(m.bands_on_side('LFS'), ['K', 'Ka', 'Q', 'V', 'W'])
        self.assertEqual(m.linearization_reference_shot, self.b.shot)

    def test_everything_works_with_the_bands_that_were_chosen(self):
        """The whole chain on a shot that lost a band that has no table, and one that borrows tables."""
        for linearization_shot in (None, self.b.shot):
            with self.subTest(linearization_shot=linearization_shot):
                m = self.load(self.a, linearization_shot=linearization_shot)
                with quiet():
                    m.post_load_init()
                    m.detector.band, m.detector.side, m.detector.sweep = 'K', 'HFS', 50
                    m.initialize_limiters('HFS', 0.0, define_init_funcs=True)
                    m.compute_all_backgrounds()
                    m.compute_current_fft()
                    m.compute_current_display()
                    m.compute_all_beatf()
                    m.compute_aggregated_delays(m.time_stamps[50])
                    profile = m.compute_profile('R (m)', m.time_stamps[50])
                for b in m.used_bands:
                    self.assertIsNotNone(m.beat_frequencies[b.side][b.name].f_probe, b.tag)
                # A band that is not used is not computed, and a W that borrows tables is
                self.assertEqual(m.beat_frequencies['LFS']['W'].f_probe is not None, linearization_shot is not None)
                self.assertGreater(profile[3].max(), 7.2e19 if linearization_shot is not None else 0)

    def test_the_model_never_asks_for_a_table_it_does_not_have(self):
        """With W excluded nothing reads W tables: the failure that started this cannot happen."""
        m = self.load(self.a)
        with quiet():
            m.post_load_init()        # backgrounds and frequency axes of every band in use
            m.detector.band, m.detector.side, m.detector.sweep = 'V', 'LFS', 30
            m.compute_sweep()
        self.assertEqual(m.current_sweep.signal_type, 'complex')


if __name__ == '__main__':
    unittest.main()
