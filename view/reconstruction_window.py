import time

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import pyqtSignal, QThread, QTimer, Qt
from PyQt5.QtGui import QIcon

from model.reconstruction import ReconstructionWorker


class ReconstructionWindow(QWidget):
    """Independent window for a single reconstruction run.

    Owns its own QThread + ReconstructionWorker.
    """

    _request_signal = pyqtSignal(object)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._finished = False

        self.setWindowTitle(f"Reconstruction - Shot {params.shot}")
        self.setWindowIcon(QIcon('reflecto-lab.png'))
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(f"Shot: {params.shot}"))

        time_text = (
            f"Time range: {params.start_time:.4f} s to "
            f"{params.end_time:.4f} s, step {params.time_step:.4f} s"
        )
        layout.addWidget(QLabel(time_text))

        outputs = []
        if params.write_private_shotfile:
            outputs.append("Private Shotfile")
        if params.write_public_shotfile:
            outputs.append("Public Shotfile")
        if params.write_hdf5:
            outputs.append(f"HDF5: {params.hdf5_destination_path}")
        layout.addWidget(QLabel("Output: " + ", ".join(outputs) if outputs else "Output: None"))

        self._status_label = QLabel("Status: Running...")
        self._status_label.setStyleSheet("color: orange; font-weight: bold;")
        layout.addWidget(self._status_label)

        self._elapsed_label = QLabel("Elapsed: 0 s")
        layout.addWidget(self._elapsed_label)

        self._close_btn = QPushButton("Close")
        self._close_btn.setEnabled(False)
        self._close_btn.clicked.connect(self.close)
        layout.addWidget(self._close_btn)

        # Elapsed time timer
        self._start_time = time.time()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._timer.start(1000)

        # Thread + worker
        self._thread = QThread()
        self._worker = ReconstructionWorker()
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(self._on_error)
        self._request_signal.connect(self._worker.reconstruct)
        self._worker.moveToThread(self._thread)
        self._thread.start()

        # Kick off reconstruction
        self._request_signal.emit(params)

    def _update_elapsed(self):
        elapsed = int(time.time() - self._start_time)
        self._elapsed_label.setText(f"Elapsed: {elapsed} s")

    def _on_finished(self):
        self._finished = True
        self._timer.stop()
        self._update_elapsed()
        self._status_label.setText("Status: Completed")
        self._status_label.setStyleSheet("color: green; font-weight: bold;")
        self._close_btn.setEnabled(True)

    def _on_error(self, error_msg):
        self._finished = True
        self._timer.stop()
        self._update_elapsed()
        self._status_label.setText(f"Status: Error - {error_msg}")
        self._status_label.setStyleSheet("color: red; font-weight: bold;")
        self._close_btn.setEnabled(True)

    def cleanup_thread(self):
        """Quit and wait for the thread."""
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(5000)

    def closeEvent(self, event):
        if not self._finished:
            reply = QMessageBox.question(
                self, "Reconstruction Running",
                "A reconstruction is still running. Close anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        self.cleanup_thread()
        event.accept()
