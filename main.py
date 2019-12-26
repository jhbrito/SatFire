import sys
from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import QObject
from PySide2.QtWidgets import QMainWindow, QFileDialog, QWidget, QLineEdit


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setMinimumHeight(600)
        self.centralWidget.setMinimumWidth(2000)

        self.gridLayout = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout.setColumnMinimumWidth(1,300)
        self.gridLayout.setColumnMinimumWidth(2,100)

        self.file_Select_Btn = QtWidgets.QPushButton(self.centralWidget)
        self.file_Input_Label = QtWidgets.QLineEdit()
        self.Ok_Btn = QtWidgets.QPushButton(self.centralWidget)
        self.file_Select_Btn.setObjectName("file_Select_Btn")
        self.file_Select_Btn.setText("Load Shape File")
        self.Ok_Btn.setObjectName("Ok_Btn")
        self.Ok_Btn.setText("OK")
        self.Ok_Btn.setDisabled(True)

        self.gridLayout.addWidget(self.file_Input_Label,1,1)
        self.gridLayout.addWidget(self.file_Select_Btn,1,2)
        self.gridLayout.addWidget(self.Ok_Btn,2,1)

        MainWindow.setCentralWidget(self.centralWidget)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        Ui_MainWindow.__init__(self)
        QMainWindow.__init__(self)

        # Initialize UI
        self.setupUi(self)
        self.file_Select_Btn.clicked.connect(self.getFilePath)
        self.Ok_Btn.clicked.connect(QtCore.QCoreApplication.instance().quit)

    def tr(self, text):
        return QObject.tr(self, text)

    def getFilePath(self):
        path_to_file, ok = QFileDialog.getOpenFileName(self, self.tr("Load File"), self.tr("~/Desktop/"), self.tr("Shape Files (*.shp)"))
        self.file_Input_Label.setText(path_to_file)
        if ok:
            self.Ok_Btn.setEnabled(True)
        else:
            self.Ok_Btn.setDisabled(True)
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())