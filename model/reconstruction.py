import logging
import rpspy
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from model.rpspy_args import build_reconstruction_kwargs

logger = logging.getLogger(__name__)


class ReconstructionWorker(QObject):
    """Handles profile reconstruction in a separate thread.

    Receives a plain ReconstructionInput data object, not the view.
    """

    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=None, **kwargs)

    @pyqtSlot(object)
    def reconstruct(self, params):
        """Perform full profile reconstruction.

        Args:
            params: ReconstructionInput dataclass with all needed values.
        """
        try:
            rpspy.full_profile_reconstruction(**build_reconstruction_kwargs(params))

            self.finished_signal.emit()
        except Exception as e:
            logger.exception("Reconstruction failed")
            self.error_signal.emit(str(e))
