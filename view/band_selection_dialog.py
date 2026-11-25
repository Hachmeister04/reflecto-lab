"""The screen shown when a shot is opened: its bands, their linearization tables, and which bands to use.

It only shows what a `ShotInspection` says (see model/shot_inspection.py, where the decisions are made):
- the bands that have a raw signal, and the entries of each band in the linearization tables;
- a checkbox for each band on each side to include it or not. All the boxes that can be checked start checked, so
  that accepting is all it takes when everything is fine. A band without entries in the linearization tables
  cannot be processed: its box is disabled and unchecked;
- the shot whose linearization tables are used, which can be changed and checked right here: the tables of
  another shot can make up for a band that has no entries in its own.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from constants import SIDES
from model.shot_inspection import USABLE, NO_TABLE

_WARNING_STYLE = "color: #c0392b;"


class BandSelectionDialog(QDialog):
    """Ask which bands of a shot to use.

    Parameters:
        inspection: ShotInspection of the shot, with the tables of its linearization shot.
        inspect: function that takes a linearization shot and returns the ShotInspection of this shot with
            the tables of that one.
    """

    def __init__(self, inspection, inspect, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ReflectoLab - bands of the shot")
        self.setModal(True)
        self._inspect = inspect
        self._inspection = inspection
        self._boxes = {}            # (band, side) -> QCheckBox of the bands that can be chosen

        layout = QVBoxLayout(self)

        # What the shot has (refilled when the tables to check change)
        self._body = QVBoxLayout()
        layout.addLayout(self._body)

        # Which linearization tables
        row = QHBoxLayout()
        row.addWidget(QLabel("Linearization shot:"))
        self._linearization_shot = QSpinBox()
        self._linearization_shot.setRange(1, 2147483647)
        self._linearization_shot.setValue(int(inspection.linearization_shot))
        row.addWidget(self._linearization_shot)
        check = QPushButton("Check tables")
        check.clicked.connect(self._on_check_tables)
        row.addWidget(check)
        row.addStretch(1)
        layout.addLayout(row)

        self._message = QLabel("")
        self._message.setStyleSheet(_WARNING_STYLE)
        self._message.setWordWrap(True)
        layout.addWidget(self._message)

        hint = QLabel('The linearization shot can also be changed later in "Linearization", under File.')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._ok = buttons.button(QDialogButtonBox.Ok)
        self._ok.setDefault(True)       # When everything is fine, which it mostly is, Enter is all it takes
        self._ok.setFocus()
        layout.addWidget(buttons)

        self._show(inspection)

    # --- What the user chose ---

    def inspection(self):
        """The ShotInspection that is being shown: it changes if other tables are checked."""
        return self._inspection

    def selection(self):
        """The bands that are checked, as a set of (band name, side)."""
        return {key for key, box in self._boxes.items() if box.isChecked()}

    # --- Content ---

    def _show(self, inspection):
        """Show an inspection: what the shot has, and a box for each band that can be used."""
        self._inspection = inspection
        self._boxes = {}

        while self._body.count():
            widget = self._body.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        report = QLabel(inspection.report())
        report.setWordWrap(True)
        report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._body.addWidget(report)

        table = QWidget()
        grid = QGridLayout(table)
        for column, text in enumerate(["Band", *SIDES, "Linearization table"]):
            label = QLabel(f"<b>{text}</b>")
            grid.addWidget(label, 0, column)

        for row, (band, states, entries) in enumerate(inspection.rows(), start=1):
            grid.addWidget(QLabel(band), row, 0)
            has_signal = False
            for column, side in enumerate(SIDES, start=1):
                state = states[side]
                if state == USABLE:
                    has_signal = True
                    box = QCheckBox("use")
                    box.setChecked(True)
                    box.toggled.connect(self._update_ok)
                    self._boxes[(band, side)] = box
                    grid.addWidget(box, row, column)
                elif state == NO_TABLE:
                    has_signal = True
                    box = QCheckBox("no table")
                    box.setChecked(False)
                    box.setEnabled(False)
                    box.setToolTip("This band has no entries in the linearization tables: it cannot be processed.")
                    grid.addWidget(box, row, column)
                else:
                    grid.addWidget(QLabel("no signal"), row, column)

            text = f"{entries} entries" if entries else "no entries available"
            label = QLabel(text)
            if has_signal and not entries:
                label.setStyleSheet(_WARNING_STYLE)
            grid.addWidget(label, row, len(SIDES) + 1)
        self._body.addWidget(table)

        if not inspection.usable_bands():
            warning = QLabel("No band can be processed with these linearization tables. "
                             "Enter another shot as the linearization shot and check its tables.")
            warning.setStyleSheet(_WARNING_STYLE)
            warning.setWordWrap(True)
            self._body.addWidget(warning)

        self._message.setText("")
        self._update_ok()

    def _update_ok(self, *_):
        """At least one band is needed."""
        self._ok.setEnabled(bool(self.selection()))

    def _on_check_tables(self):
        """Look at the tables of the shot that was entered."""
        try:
            inspection = self._inspect(self._linearization_shot.value())
        except (ValueError, OSError) as e:
            self._message.setText(f"Could not check the tables: {e}")
            return
        self._show(inspection)
