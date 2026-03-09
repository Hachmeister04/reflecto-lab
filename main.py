import sys
import logging
from PyQt5.QtWidgets import QApplication
from controller.app_controller import AppController

logging.basicConfig(level=logging.INFO)

# Profiling output to file
_perf_handler = logging.FileHandler('profiling.log', mode='w')
_perf_handler.setFormatter(logging.Formatter('%(message)s'))
for _name in ('controller.app_controller', 'model.shot_model'):
    _logger = logging.getLogger(_name)
    _logger.addHandler(_perf_handler)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = AppController()
    controller.view.show()
    sys.exit(app.exec())
