import sys
import logging
from PyQt5.QtWidgets import QApplication
from controller.app_controller import AppController

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = AppController()
    controller.view.show()
    sys.exit(app.exec())
