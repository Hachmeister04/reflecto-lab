from PyQt5.QtWidgets import QMessageBox


def show_warning():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Warning")
    msg.setText("Raw data not found for the selected shot \n OR \n Selected folder does not contain valid raw data.")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()
