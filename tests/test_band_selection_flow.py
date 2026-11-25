"""The band selection dialog and the flow that opens a shot, run on a minimal stand-in for Qt.

The real dialog and controller code run, with fake widgets that remember what was done to them, and a real
model on synthetic shots. This does not replace looking at the window, but it checks the logic: what is
offered, what is checked, what happens when the user accepts, cancels or checks other tables.
Run from the reflecto-lab folder::

    python3 -m unittest discover -s tests -v
"""
import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
RPSPY = os.environ.get("RPSPY_PATH", os.path.join(os.path.dirname(LAB), "rpspy"))
sys.path[:0] = [LAB, RPSPY, os.path.join(RPSPY, "tests")]

try:
    from synthetic_shot import make_synthetic_shot, install_environment, _stub_missing_modules
except ImportError as e:  # pragma: no cover
    raise unittest.SkipTest(f"synthetic shot of rpspy not found in {RPSPY}: {e}")


# ------------------------------------------------------------------------------------------------------
# A stand-in for the parts of Qt that are used

class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class Widget:
    def __init__(self, *args, **kwargs):
        self.enabled = True
        self.text = args[0] if args and isinstance(args[0], str) else ''
        self.deleted = False
        self.tooltip = ''

    def setParent(self, parent):
        pass

    def deleteLater(self):
        self.deleted = True

    def setEnabled(self, enabled):
        self.enabled = enabled

    def setDefault(self, default):
        self.is_default = default

    def setFocus(self):
        self.has_focus = True

    def isEnabled(self):
        return self.enabled

    def setText(self, text):
        self.text = text

    def setToolTip(self, text):
        self.tooltip = text

    def __getattr__(self, name):         # setWordWrap, setStyleSheet, setModal, ...
        if name.startswith('__'):
            raise AttributeError(name)
        return mock.MagicMock()


class QCheckBox(Widget):
    created = []        # every box made, to look at the ones that are not offered as a choice

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked = False
        self.toggled = Signal()
        QCheckBox.created.append(self)

    def setChecked(self, checked):
        changed = checked != self.checked
        self.checked = checked
        if changed:
            self.toggled.emit(checked)

    def isChecked(self):
        return self.checked


class QSpinBox(Widget):
    _value = 0

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value


class QPushButton(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clicked = Signal()


class Item:
    def __init__(self, widget):
        self._widget = widget

    def widget(self):
        return self._widget if isinstance(self._widget, Widget) else None


class Layout:
    def __init__(self, *args, **kwargs):
        self.items = []

    def addWidget(self, widget, *args, **kwargs):
        self.items.append(widget)

    def addLayout(self, layout, *args, **kwargs):
        self.items.append(layout)

    def addStretch(self, *args):
        pass

    def count(self):
        return len(self.items)

    def takeAt(self, index):
        return Item(self.items.pop(index))


class QDialog(Widget):
    accept_on_exec = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = False

    def accept(self):
        self.result = True

    def reject(self):
        self.result = False

    def exec(self):
        self.executed = True
        return self.accept_on_exec


class QDialogButtonBox(Widget):
    Ok, Cancel = 1, 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted, self.rejected = Signal(), Signal()
        self._buttons = {}

    def button(self, which):
        return self._buttons.setdefault(which, Widget())


def fake_qt():
    core = types.ModuleType('PyQt5.QtCore')
    core.Qt = types.SimpleNamespace(TextSelectableByMouse=1, WaitCursor=2, WindowModal=3, DashLine=4)
    core.QObject = type('QObject', (), {'__init__': lambda self, *a, **k: None})
    core.QTimer = mock.MagicMock()
    widgets = types.ModuleType('PyQt5.QtWidgets')
    for name, cls in dict(QCheckBox=QCheckBox, QSpinBox=QSpinBox, QPushButton=QPushButton, QDialog=QDialog,
                          QDialogButtonBox=QDialogButtonBox, QLabel=Widget, QWidget=Widget, QGridLayout=Layout,
                          QHBoxLayout=Layout, QVBoxLayout=Layout).items():
        setattr(widgets, name, cls)
    for name in ('QApplication', 'QFileDialog', 'QMessageBox', 'QProgressDialog'):
        setattr(widgets, name, mock.MagicMock())
    pyqt = types.ModuleType('PyQt5')
    pyqt.QtCore, pyqt.QtWidgets = core, widgets
    modules = {'PyQt5': pyqt, 'PyQt5.QtCore': core, 'PyQt5.QtWidgets': widgets}
    # The views that are not under test
    for name in ('view.main_window', 'view.reconstruction_window', 'view.parameter_panels', 'view.plot_renderers'):
        modules[name] = mock.MagicMock()
    return modules


class Param:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, value, blockSignal=None):
        self._value = value


class Group:
    def __init__(self, **children):
        self.children = children

    def child(self, name):
        return self.children[name]


# ------------------------------------------------------------------------------------------------------

class FlowTest(unittest.TestCase):
    """Shot A has W recorded but no table entries for it. Shot B has tables for every band."""

    @classmethod
    def setUpClass(cls):
        _stub_missing_modules()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.a = make_synthetic_shot(cls._tmp.name, shot=45070, n_sweeps=200, with_w=True, no_table_bands=('W',), seed=0)
        cls.b = make_synthetic_shot(cls._tmp.name, shot=45071, n_sweeps=200, with_w=True, seed=1)
        install_environment(cls.a)

        for name in [n for n in sys.modules if n.split('.')[0] in ('view', 'controller', 'utils')]:
            sys.modules.pop(name)
        cls._patch = mock.patch.dict(sys.modules, fake_qt())
        cls._patch.start()
        cls.dialog_module = importlib.import_module('view.band_selection_dialog')
        cls.controller_module = importlib.import_module('controller.app_controller')
        from model.shot_model import ShotModel
        cls.ShotModel = ShotModel

    @classmethod
    def tearDownClass(cls):
        cls._patch.stop()
        for name in [n for n in sys.modules if n.split('.')[0] in ('view', 'controller', 'utils')]:
            sys.modules.pop(name, None)
        cls._tmp.cleanup()

    def inspect(self, shot, linearization_shot=None):
        return self.ShotModel().inspect_shot(shot.shot, shot.shot_dir, linearization_shot)

    def make_dialog(self, shot, linearization_shot=None):
        base = self.ShotModel()
        inspection = base.inspect_shot(shot.shot, shot.shot_dir, linearization_shot)
        calls = []

        def inspect(reference):
            calls.append(reference)
            return base.inspect_shot(shot.shot, shot.shot_dir, reference)

        dialog = self.dialog_module.BandSelectionDialog(inspection, inspect)
        dialog.calls = calls
        return dialog

    # --- the dialog ---

    def test_a_box_for_each_band_that_can_be_used(self):
        dialog = self.make_dialog(self.a)
        offered = set(dialog._boxes)
        self.assertEqual(len(offered), 8)
        self.assertNotIn(('W', 'LFS'), offered)
        self.assertEqual(dialog.selection(), offered)           # all checked
        self.assertTrue(dialog._ok.isEnabled())

    def test_every_box_that_can_be_checked_starts_checked(self):
        for linearization_shot in (None, self.b.shot):
            with self.subTest(linearization_shot=linearization_shot):
                dialog = self.make_dialog(self.a, linearization_shot)
                self.assertTrue(dialog._boxes)
                self.assertTrue(all(box.isChecked() and box.isEnabled() for box in dialog._boxes.values()))
                self.assertEqual(dialog.selection(), dialog.inspection().default_selection())

    def test_only_the_bands_that_cannot_be_processed_are_not_checked(self):
        QCheckBox.created.clear()
        self.make_dialog(self.a)
        unchecked = [box for box in QCheckBox.created if not box.isChecked()]
        self.assertEqual([box.text for box in unchecked], ["no table"])      # W-LFS
        self.assertFalse(unchecked[0].isEnabled())
        self.assertEqual(sum(box.isChecked() for box in QCheckBox.created), 8)

    def test_checking_other_tables_checks_everything_again(self):
        dialog = self.make_dialog(self.a)
        dialog._boxes[('K', 'HFS')].setChecked(False)
        dialog._linearization_shot.setValue(self.b.shot)
        dialog._on_check_tables()
        self.assertTrue(all(box.isChecked() for box in dialog._boxes.values()))
        self.assertEqual(len(dialog.selection()), 9)

    def test_enter_accepts_when_everything_is_fine(self):
        dialog = self.make_dialog(self.a)
        self.assertTrue(dialog._ok.is_default)
        self.assertTrue(dialog._ok.has_focus)
        self.assertTrue(dialog._ok.isEnabled())

    def test_the_user_can_leave_bands_out(self):
        dialog = self.make_dialog(self.a)
        dialog._boxes[('Q', 'HFS')].setChecked(False)
        dialog._boxes[('V', 'LFS')].setChecked(False)
        self.assertEqual(len(dialog.selection()), 6)
        self.assertNotIn(('Q', 'HFS'), dialog.selection())

    def test_at_least_one_band_is_needed(self):
        dialog = self.make_dialog(self.a)
        for box in dialog._boxes.values():
            box.setChecked(False)
        self.assertEqual(dialog.selection(), set())
        self.assertFalse(dialog._ok.isEnabled())
        dialog._boxes[('K', 'HFS')].setChecked(True)
        self.assertTrue(dialog._ok.isEnabled())

    def test_a_shot_without_usable_bands_cannot_be_accepted(self):
        dialog = self.make_dialog(self.a, linearization_shot=99)      # no tables for shot 99
        self.assertEqual(dialog._boxes, {})
        self.assertFalse(dialog._ok.isEnabled())

    def test_checking_the_tables_of_another_shot_offers_the_band(self):
        dialog = self.make_dialog(self.a)
        self.assertNotIn(('W', 'LFS'), dialog._boxes)
        dialog._linearization_shot.setValue(self.b.shot)
        dialog._on_check_tables()                                       # the user presses "Check tables"
        self.assertEqual(dialog.calls, [self.b.shot])
        self.assertIn(('W', 'LFS'), dialog._boxes)
        self.assertTrue(dialog._boxes[('W', 'LFS')].isChecked())
        self.assertEqual(dialog.inspection().linearization_shot, self.b.shot)
        self.assertEqual(len(dialog.selection()), 9)

    def test_what_was_shown_before_is_removed(self):
        dialog = self.make_dialog(self.a)
        before = list(dialog._body.items)
        dialog._on_check_tables()
        self.assertTrue(all(w.deleted for w in before))
        self.assertEqual(len(dialog._body.items), len(before))          # the report and the table, again

    def test_a_failure_to_check_is_shown_not_raised(self):
        base = self.ShotModel()
        inspection = base.inspect_shot(self.a.shot, self.a.shot_dir)

        def failing(reference):
            raise ValueError("nope")

        dialog = self.dialog_module.BandSelectionDialog(inspection, failing)
        dialog._on_check_tables()
        self.assertIn("nope", dialog._message.text)
        self.assertEqual(dialog.inspection(), inspection)               # nothing changed

    def test_the_report_is_shown(self):
        dialog = self.make_dialog(self.a)
        texts = [w.text for w in dialog._body.items if isinstance(w, Widget)]
        self.assertTrue(any("W band - no entries available" in t for t in texts))

    # --- opening a shot ---

    def controller(self, choose=None):
        """A controller without its window, with a File panel and a model."""
        module = self.controller_module
        ctl = module.AppController.__new__(module.AppController)
        ctl.model = self.ShotModel()
        ctl.panels = types.SimpleNamespace(file=Group(Open=Param(''), Shot=Param(0), Linearization=Param(0)))
        ctl.view = mock.MagicMock()
        ctl._h_shot_open, ctl._h_shot_number, ctl._h_linearization_shot = (lambda: None,) * 3
        ctl.chosen = []
        if choose is not None:
            def _choose(inspection):
                ctl.chosen.append(inspection)
                return choose(inspection)
            ctl._choose_bands = _choose
        patcher = mock.patch.object(module, 'show_warning')
        ctl.warning = patcher.start()
        self.addCleanup(patcher.stop)
        patch_path = mock.patch.object(module, 'get_path_from_shot', lambda shot: {self.a.shot: self.a.shot_dir, self.b.shot: self.b.shot_dir}.get(shot, '/nowhere/%d' % shot))
        patch_path.start()
        self.addCleanup(patch_path.stop)
        return ctl

    def everything(self, inspection):
        return inspection, inspection.default_selection()

    def test_opening_a_shot_loads_the_bands_that_can_be_processed(self):
        ctl = self.controller(self.everything)
        ctl.panels.file.child('Shot').setValue(self.a.shot)
        loading = mock.MagicMock()
        self.assertTrue(ctl._select_and_load('shot', loading))
        m = ctl.model
        self.assertEqual((m.shot, m.linearization_reference_shot), (self.a.shot, self.a.shot))
        self.assertEqual(len(m.used_bands), 8)
        self.assertNotIn('W', m.used_band_names)
        # The File panel says what is loaded
        file = ctl.panels.file
        self.assertEqual((file.child('Open').value(), file.child('Shot').value(), file.child('Linearization').value()),
                         (self.a.shot_dir, self.a.shot, self.a.shot))
        # The user was shown the shot before anything was loaded, with the busy message out of the way
        self.assertEqual(ctl.chosen[0].shot, self.a.shot)
        loading.hide.assert_called()
        loading.show.assert_called()
        ctl.warning.assert_not_called()

    def test_opening_a_folder(self):
        ctl = self.controller(self.everything)
        ctl.panels.file.child('Open').setValue(self.b.shot_dir)
        self.assertTrue(ctl._select_and_load('open', mock.MagicMock()))
        self.assertEqual(ctl.model.shot, self.b.shot)
        self.assertEqual(len(ctl.model.used_bands), 9)
        self.assertEqual(ctl.panels.file.child('Shot').value(), self.b.shot)

    def test_the_user_chooses_the_bands(self):
        ctl = self.controller(lambda insp: (insp, {('K', 'HFS'), ('K', 'LFS')}))
        ctl.panels.file.child('Shot').setValue(self.a.shot)
        self.assertTrue(ctl._select_and_load('shot', mock.MagicMock()))
        self.assertEqual({(b.name, b.side) for b in ctl.model.used_bands}, {('K', 'HFS'), ('K', 'LFS')})

    def test_cancelling_loads_nothing(self):
        ctl = self.controller(lambda insp: None)
        ctl.panels.file.child('Shot').setValue(self.a.shot)
        self.assertFalse(ctl._select_and_load('shot', mock.MagicMock()))
        self.assertEqual((ctl.model.shot, ctl.model.used_bands), (0, []))
        self.assertEqual(ctl.panels.file.child('Shot').value(), 0)      # back to what is loaded: nothing

    def test_cancelling_keeps_the_shot_that_was_loaded(self):
        ctl = self.controller(self.everything)
        ctl.panels.file.child('Shot').setValue(self.a.shot)
        self.assertTrue(ctl._select_and_load('shot', mock.MagicMock()))
        ctl._choose_bands = lambda insp: None
        ctl.panels.file.child('Shot').setValue(self.b.shot)
        self.assertFalse(ctl._select_and_load('shot', mock.MagicMock()))
        self.assertEqual(ctl.model.shot, self.a.shot)
        self.assertEqual(ctl.panels.file.child('Shot').value(), self.a.shot)

    def test_a_shot_without_data_is_a_warning_and_the_dialog_is_not_shown(self):
        ctl = self.controller(self.everything)
        ctl.panels.file.child('Shot').setValue(12345)
        self.assertFalse(ctl._select_and_load('shot', mock.MagicMock()))
        ctl.warning.assert_called_once()
        self.assertEqual(ctl.chosen, [])
        self.assertEqual(ctl.model.shot, 0)

    def test_a_folder_that_is_not_a_shot_is_a_warning(self):
        ctl = self.controller(self.everything)
        ctl.panels.file.child('Open').setValue('/some/folder')
        self.assertFalse(ctl._select_and_load('open', mock.MagicMock()))
        ctl.warning.assert_called_once()
        self.assertEqual(ctl.panels.file.child('Open').value(), '')

    def test_changing_the_linearization_shot_reopens_the_shot_with_other_tables(self):
        ctl = self.controller(self.everything)
        ctl.panels.file.child('Shot').setValue(self.a.shot)
        ctl._select_and_load('shot', mock.MagicMock())
        self.assertNotIn('W', ctl.model.used_band_names)

        ctl.panels.file.child('Linearization').setValue(self.b.shot)     # the user types another shot
        self.assertTrue(ctl._select_and_load('linearization', mock.MagicMock()))
        m = ctl.model
        self.assertEqual((m.shot, m.linearization_reference_shot), (self.a.shot, self.b.shot))
        self.assertIn(('W', 'LFS', 'complex'), [tuple(b) for b in m.used_bands])
        self.assertEqual(ctl.chosen[-1].linearization_shot, self.b.shot)
        self.assertEqual(ctl.chosen[-1].shot, self.a.shot)
        self.assertEqual(ctl.panels.file.child('Linearization').value(), self.b.shot)

    def test_cancelling_a_change_of_linearization_shot_restores_it(self):
        ctl = self.controller(self.everything)
        ctl.panels.file.child('Shot').setValue(self.a.shot)
        ctl._select_and_load('shot', mock.MagicMock())
        ctl._choose_bands = lambda insp: None
        ctl.panels.file.child('Linearization').setValue(self.b.shot)
        self.assertFalse(ctl._select_and_load('linearization', mock.MagicMock()))
        self.assertEqual(ctl.model.linearization_reference_shot, self.a.shot)
        self.assertEqual(ctl.panels.file.child('Linearization').value(), self.a.shot)

    def test_the_linearization_shot_does_nothing_before_a_shot_is_open(self):
        ctl = self.controller(self.everything)
        ctl.panels.file.child('Linearization').setValue(self.b.shot)
        self.assertFalse(ctl._select_and_load('linearization', mock.MagicMock()))
        self.assertEqual(ctl.chosen, [])
        self.assertEqual(ctl.panels.file.child('Linearization').value(), 0)

    def test_the_linearization_shot_chosen_in_the_dialog_is_the_one_used(self):
        """The user checks the tables of another shot inside the dialog, then accepts."""
        module = self.controller_module
        ctl = self.controller()
        ctl.panels.file.child('Shot').setValue(self.a.shot)
        real_dialog = module.BandSelectionDialog

        class Accepting(real_dialog):
            def exec(self):
                self._linearization_shot.setValue(ctl_self.b.shot)
                self._on_check_tables()
                return True

        ctl_self = self
        with mock.patch.object(module, 'BandSelectionDialog', Accepting):
            self.assertTrue(ctl._select_and_load('shot', mock.MagicMock()))
        self.assertEqual(ctl.model.linearization_reference_shot, self.b.shot)
        self.assertIn('W', ctl.model.used_band_names)
        self.assertEqual(ctl.panels.file.child('Linearization').value(), self.b.shot)

    def test_the_busy_cursor_is_not_shown_over_the_dialog(self):
        module = self.controller_module
        ctl = self.controller()
        ctl.view = None
        inspection = self.inspect(self.a)
        app = module.QApplication
        app.reset_mock()
        order = []
        app.restoreOverrideCursor.side_effect = lambda: order.append('restore')
        app.setOverrideCursor.side_effect = lambda cursor: order.append('set')
        with mock.patch.object(module.BandSelectionDialog, 'exec', lambda self: order.append('exec') or True):
            inspection_back, selected = ctl._choose_bands(inspection)
        self.assertEqual(order, ['restore', 'exec', 'set'])
        self.assertEqual(selected, inspection.default_selection())
        with mock.patch.object(module.BandSelectionDialog, 'exec', lambda self: False):
            self.assertIsNone(ctl._choose_bands(inspection))

    def test_the_reconstruction_is_given_the_linearization_shot_that_was_chosen(self):
        ctl = self.controller(lambda insp: (self.ShotModel().inspect_shot(insp.shot, insp.file_path, self.b.shot),
                                            {('K', 'HFS'), ('W', 'LFS')}))
        ctl.panels.file.child('Shot').setValue(self.a.shot)
        ctl._select_and_load('shot', mock.MagicMock())
        m = ctl.model
        from model.state import ReconstructionInput
        from model.rpspy_args import build_reconstruction_kwargs
        params = ReconstructionInput(shot=m.shot, file_path=m.file_path, spect_params=m.spect_params, filters=m.filters,
                                     frequency_exclusions=m.frequency_exclusions, spectrogram_masks=m.spectrogram_masks,
                                     used_bands=list(m.used_bands), linearization_shot=m.linearization_reference_shot)
        kwargs = build_reconstruction_kwargs(params)
        self.assertEqual(kwargs['shot'], self.a.shot)
        self.assertEqual(kwargs['shot_linearization'], self.b.shot)
        self.assertEqual({(n, s) for n, s, _ in kwargs['used_bands']}, {('K', 'HFS'), ('W', 'LFS')})


if __name__ == '__main__':
    unittest.main()
