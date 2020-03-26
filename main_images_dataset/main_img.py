import sys
from Interface_img import MainWindow
from PySide2 import QtCore, QtWidgets

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    