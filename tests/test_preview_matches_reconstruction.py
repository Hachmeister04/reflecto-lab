"""What reflecto-lab shows for an instant must be what "Reconstruct Shot" produces for it.

The preview (ShotModel) and the reconstruction (the arguments the Reconstruct button hands to
rpspy.full_profile_reconstruction) are run on the same settings of a synthetic shot and compared.
No AUG data and no Qt are needed. Run from the reflecto-lab folder::

    python3 -m unittest discover -s tests -v

rpspy is looked for next to this repository, or in $RPSPY_PATH.
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

BANDS = ('K', 'Ka', 'Q', 'V')
SIDES = ('HFS', 'LFS')


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class PreviewMatchesReconstruction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.shot = make_synthetic_shot(cls._tmp.name)
        install_environment(cls.shot)
        import rpspy
        from model.shot_model import ShotModel
        from model import state
        from model.rpspy_args import build_reconstruction_kwargs
        cls.rpspy, cls.ShotModel, cls.state = rpspy, ShotModel, state
        cls.build_kwargs = staticmethod(build_reconstruction_kwargs)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # --------------------------------------------------------------------------------------------
    # Scenarios: each one configures a model the way a user would through the panels

    def new_model(self, burst, start, end, step, nperseg=128, noverlap=96, nfft=512):
        m = self.ShotModel()
        with quiet():
            m.load_shot_from_path(self.shot.shot_dir)
        m.detector.burst_size = burst
        m.reconstruction_start_time, m.reconstruction_end_time, m.reconstruction_time_step = start, end, step
        for side in SIDES:
            for band in BANDS:
                sp = m.spect_params[side][band]
                sp.nperseg, sp.noverlap, sp.nfft = nperseg, noverlap, nfft
                sp.subtract_dispersion = band == 'V'
                sp.subtract_background = False
                sp.background_sweep, sp.background_burst_size = 60, 3
                m.filters[side][band].low, m.filters[side][band].high = 0.3e6, 6e6
        with quiet():
            m.post_load_init()
        return m

    def finish(self, m):
        """What the controller does once the user is done with the panels."""
        with quiet():
            m.initialize_limiters('HFS', 0.0, define_init_funcs=True)
            m.initialize_limiters('LFS', 0.0, define_init_funcs=True)
            m.compute_all_backgrounds()

    def scenario_default(self):
        m = self.new_model(burst=5, start=0.004, end=0.012, step=0.001)
        for side, band in (('HFS', 'Q'), ('LFS', 'K')):
            m.spect_params[side][band].subtract_background = True
        for band in SIDES:
            m.filters['LFS'][band if band in BANDS else 'K'].low = 0.0
        for band in BANDS:
            m.filters['LFS'][band].low, m.filters['LFS'][band].high = 0.0, 19e6
        m.spect_params['HFS']['Ka'].subtract_dispersion = True
        FE, SM = self.state.FrequencyExclusion, self.state.SpectrogramMask
        m.frequency_exclusions['HFS'] += [FE(38e9, 40e9, 0.0, 0.0065, True), FE(44e9, 45e9, 0.0, 10.0, False)]
        m.frequency_exclusions['LFS'] += [FE(28e9, 29e9, 0.0, 10.0, True)]
        m.spectrogram_masks['HFS']['K'] += [SM(0.0, 0.0125, 20e9, 23e9, 0.0, 2e6, True)]
        m.spectrogram_masks['LFS']['Ka'] += [SM(0.0, 10.0, 30e9, 31e9, 0.0, 19e6, True),
                                             SM(0.0, 10.0, 33e9, 34e9, 0.0, 19e6, False)]
        self.finish(m)
        return m, (np.array([0.0, 10.0]), np.array([3e19, 3e19]))

    def scenario_no_overlap_files_and_per_band_options(self):
        m = self.new_model(burst=1, start=0.004, end=0.010, step=0.0005)
        m.spect_params['HFS']['K'].nperseg, m.spect_params['HFS']['K'].noverlap, m.spect_params['HFS']['K'].nfft = 64, 48, 256
        m.spect_params['LFS']['Q'].subtract_dispersion = True
        m.spect_params['HFS']['K'].subtract_background = True
        m.spect_params['HFS']['K'].background_sweep, m.spect_params['HFS']['K'].background_burst_size = 100, 5
        m.filters['HFS']['Ka'].low, m.filters['HFS']['Ka'].high = 0.55e6, 2.0e6    # edge right next to the peak
        iv = m.init_values['HFS']
        iv.type, iv.file.name = 'From file', 'init.csv (loaded)'
        iv.file.time, iv.file.position = np.array([0.0, 10.0]), np.array([1.03, 1.07])
        m.init_values['LFS'].type, m.init_values['LFS'].custom_value = 'Custom', 2.19
        self.finish(m)
        return m, (None, None)

    def scenario_noisy_hfs_and_overlapping_bursts(self):
        # Q-HFS is quiet from sweep 450 on: the HFS profile has to be discarded above 36 GHz
        m = self.new_model(burst=3, start=0.016, end=0.020, step=0.0001)
        self.finish(m)
        return m, (np.array([0.0, 10.0]), np.array([5e19, 5e19]))

    # --------------------------------------------------------------------------------------------

    def reconstruct(self, m, cutoff):
        p = m.reconstruction_start_time, m.reconstruction_end_time, m.reconstruction_time_step
        params = self.state.ReconstructionInput(
            shot=m.shot, file_path=m.file_path, spect_params=m.spect_params, filters=m.filters,
            frequency_exclusions=m.frequency_exclusions, spectrogram_masks=m.spectrogram_masks,
            burst_size=m.detector.burst_size, start_time=p[0], end_time=p[1], time_step=p[2],
            density_cutoff_times=cutoff[0], density_cutoff_values=cutoff[1],
            write_hdf5=False, hdf5_destination_path=None,
            get_init_lfs=lambda time: m.get_init('LFS', time),
            get_init_hfs=lambda time: m.get_init('HFS', time),
        )
        kwargs = self.build_kwargs(params)
        kwargs['return_profiles'] = True
        with quiet():
            profiles = self.rpspy.full_profile_reconstruction(**kwargs)
            per_band = {}
            for side in SIDES:
                for band in BANDS:
                    opts = kwargs['spectrogram_options'][side][band]
                    lower, upper = kwargs['filters'][side][band]
                    per_band[(side, band)] = self.rpspy.batch_processing(
                        kwargs['shot'], band, side, 'complex' if band == 'V' else 'real',
                        kwargs['start_time'], kwargs['end_time'], kwargs['time_step'], kwargs['burst'],
                        lower, upper, kwargs['subtract_background_on_bands'], kwargs['subtract_dispersion_on_bands'],
                        kwargs['shotfile_dir'], kwargs['shot_linearization'], kwargs['linearization_shotfile_dir'],
                        kwargs['sweep_linearization'], nperseg=opts['nperseg'], nfft=opts['nfft'], noverlap=opts['noverlap'],
                        background_sweep=kwargs['background_sweeps'][side][band],
                        background_burst_size=kwargs['background_burst_size'][side][band],
                        masks=kwargs['spectrogram_masks'][side][band])
        return profiles, per_band

    def preview(self, m, sweep):
        """What the controller computes when the user moves to `sweep`."""
        t = m.time_stamps[sweep]
        m.detector.sweep = sweep
        with quiet():
            m.initialize_limiters(m.detector.side, t)
            m.compute_current_fft()
            m.compute_current_display()
            m.compute_all_beatf()
            m.compute_aggregated_delays(t)
            return m.compute_profile('R (m)', t)

    def check(self, m, cutoff):
        (times, ne, r_hfs, r_lfs), per_band = self.reconstruct(m, cutoff)

        grid = m.reconstruction_sweeps()
        np.testing.assert_array_equal(m.time_stamps[grid], times)     # same instants as the reconstruction

        instants = sorted({0, 1, len(grid) // 2, len(grid) - 1})
        for i in instants:
            preview = self.preview(m, grid[i])

            for (side, band), (_, f_probe, group_delay) in per_band.items():
                with self.subTest(instant=i, side=side, band=band, what='group delay'):
                    bf = m.beat_frequencies[side][band]
                    np.testing.assert_array_equal(bf.f_probe, f_probe)
                    np.testing.assert_allclose(bf.y_beat_time, group_delay[i], rtol=1e-9, atol=0)

            t = times[i]
            for name, r_prev, ne_prev, r_rec in (('HFS', preview[0], preview[1], r_hfs[i]),
                                                 ('LFS', preview[2], preview[3], r_lfs[i])):
                with self.subTest(instant=i, side=name, what='profile'):
                    where = np.searchsorted(ne, ne_prev)
                    np.testing.assert_array_equal(ne[where], ne_prev)   # same density grid

                    kept = np.flatnonzero(np.isfinite(r_rec))
                    self.assertTrue(np.isin(kept, where).all(), 'the reconstruction has points the preview does not')
                    shown = np.isin(where, kept)
                    np.testing.assert_allclose(r_prev[shown], r_rec[where[shown]], rtol=1e-9, atol=0)

                    # What the preview shows and the reconstruction drops is only what the cutoff removes
                    if cutoff[0] is None:
                        self.assertTrue(shown.all())
                    else:
                        limit = np.interp(t, *cutoff)
                        self.assertTrue((ne_prev[~shown] >= limit).all())

        return r_hfs

    # --------------------------------------------------------------------------------------------

    def test_default_settings_with_masks_exclusions_background_and_dispersion(self):
        self.check(*self.scenario_default())

    def test_no_burst_averaging_file_initialization_and_other_spectrogram_options(self):
        self.check(*self.scenario_no_overlap_files_and_per_band_options())

    def test_noisy_hfs_and_overlapping_bursts(self):
        m, cutoff = self.scenario_noisy_hfs_and_overlapping_bursts()
        r_hfs = self.check(m, cutoff)
        f = np.linspace(0, 75e9, 151)
        # Quiet Q-HFS: no HFS profile above 36 GHz, where there is something to show it is NaN
        self.assertTrue(np.isnan(r_hfs[:, f >= 36e9]).all())
        self.assertTrue(np.isfinite(r_hfs[:, f < 30e9]).any())

    def test_instant_status_in_reconstruction_grid(self):
        m, _ = self.scenario_default()
        grid = m.reconstruction_sweeps()
        self.assertEqual(len(np.unique(np.diff(grid))), 1)
        self.assertTrue(set(grid) <= set(range(len(m.time_stamps))))


    def test_snapping_to_the_reconstruction_grid(self):
        m, _ = self.scenario_default()
        grid = m.reconstruction_sweeps()
        step = int(grid[1] - grid[0])
        on, off = int(grid[3]), int(grid[3]) + 1

        # Already on the grid: stays
        self.assertEqual(m.snap_to_reconstruction_grid(on, previous=on - 1), on)
        # One sweep at a time: steps through the grid in the direction of the move, never sticks
        self.assertEqual(m.snap_to_reconstruction_grid(on + 1, previous=on), on + step)
        self.assertEqual(m.snap_to_reconstruction_grid(on - 1, previous=on), on - step)
        shown = on
        for _ in range(3):
            shown = m.snap_to_reconstruction_grid(shown + 1, previous=shown)
        self.assertEqual(shown, on + 3 * step)
        # A jump goes to the nearest grid sweep, and so does turning snapping on in place
        self.assertEqual(m.snap_to_reconstruction_grid(on + 3 * step + 2, previous=on), on + 3 * step)
        self.assertEqual(m.snap_to_reconstruction_grid(off, previous=off), on)
        self.assertEqual(m.snap_to_reconstruction_grid(on + step - 1, previous=on + step - 1), on + step)
        # Beyond the ends: the first and last grid sweep
        self.assertEqual(m.snap_to_reconstruction_grid(int(grid[-1]) + 1, previous=int(grid[-1])), int(grid[-1]))
        self.assertEqual(m.snap_to_reconstruction_grid(int(grid[0]) - 1, previous=int(grid[0])), int(grid[0]))


if __name__ == '__main__':
    unittest.main()
