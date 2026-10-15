from contextlib import contextmanager

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog


def show_warning():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Warning")
    msg.setText("Raw data not found for the selected shot \n OR \n Selected folder does not contain valid raw data.")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


@contextmanager
def loading_indicator(parent, text):
    """Show that the program is busy: a message with a busy bar and a wait cursor.

    The work happens in the interface thread, so the bar does not animate. Update the message
    with ``setLabelText`` between the steps of the work, and ``hide`` it before showing any
    other dialog.
    """
    dialog = QProgressDialog(text, None, 0, 0, parent)
    dialog.setWindowTitle("ReflectoLab")
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.show()
    QApplication.setOverrideCursor(Qt.WaitCursor)
    QApplication.processEvents()
    try:
        yield dialog
    finally:
        QApplication.restoreOverrideCursor()
        dialog.close()
