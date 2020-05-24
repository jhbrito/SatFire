import sys
from PySide2 import QtCore, QtWidgets
from Display_ShapeFile_img import DisplayAllShapes
from get_shapes_between_x_and_y_img import ProcesseShapes
from get_image_map_img import GetImageMap

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setMinimumHeight(600)
        self.centralWidget.setMinimumWidth(2000)

        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.horizontalLayout1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout6 = QtWidgets.QHBoxLayout()
        self.verticalLayout2 = QtWidgets.QVBoxLayout()


        self.file_Select_Btn = QtWidgets.QPushButton()
        self.file_Input_Label = QtWidgets.QLineEdit()
        self.rb2 = QtWidgets.QRadioButton('Generate Images:')
        self.Ok_Btn = QtWidgets.QPushButton()
        self.Exit_Btn = QtWidgets.QPushButton()

        self.file_Select_Btn.setObjectName("file_Select_Btn")
        self.file_Select_Btn.setText("Load Shape File")
        self.Ok_Btn.setObjectName("Ok_Btn")
        self.Ok_Btn.setText("OK")
        self.Ok_Btn.setDisabled(True)

        self.Exit_Btn.setObjectName("Exit_Btn")
        self.Exit_Btn.setText("Exit")


        self.horizontalLayout1.addWidget(self.file_Input_Label)
        self.horizontalLayout1.addWidget(self.file_Select_Btn)
        self.horizontalLayout4.addWidget(self.rb2, 50, QtCore.Qt.AlignLeft)
        self.horizontalLayout6.addWidget(self.Ok_Btn)
        self.horizontalLayout6.addWidget(self.Exit_Btn)

        self.verticalLayout.addLayout(self.horizontalLayout1)

        self.verticalLayout2.addLayout(self.horizontalLayout4)
        self.verticalLayout.addLayout(self.horizontalLayout6)


        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        Ui_MainWindow.__init__(self)
        QtWidgets.QMainWindow.__init__(self)
        self.path = ''
        self.dirname = ''

        # Initialize UI
        self.setupUi(self)
        self.file_Select_Btn.clicked.connect(self.getFilePath)
        self.Exit_Btn.clicked.connect(QtCore.QCoreApplication.instance().quit)   #close application
        self.Ok_Btn.clicked.connect(self.clickOkBtn)
        self.rb2.setChecked(True)
        self.rb2.toggled.connect(lambda: self.rb_clicked())

    def tr(self, text):
        return QtCore.QObject.tr(self, text)

    def getFilePath(self):
        self.path, self.ok = QtWidgets.QFileDialog.getOpenFileName(self, self.tr("Load File"), self.tr("~/Desktop/"), self.tr("Shape Files (*.shp)"))
        self.file_Input_Label.setText(self.path)
        self.enable_OkBtn()

    def getSavePath(self):
        self.dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory", '~/')
        self.dirname = self.dirname + '/new_file.shp'
        self.file_save_path.setText(self.dirname)
        self.enable_OkBtn()

    def clickOkBtn(self):

        self.Ok_Btn.setDisabled(True)

        # PROCESS SHAPEFILE AND GENERATE DATASET
        if self.rb2.isChecked():
            file2 = ProcesseShapes(self.path)
            file2.process_file()

        self.Ok_Btn.setEnabled(True)

    def rb_clicked(self):
        #on radio buttons click, the interface layout changes
        if self.rb2.isChecked():
            self.file_save_path.setVisible(False)
            self.file_save_path_Btn.setVisible(False)

        self.enable_OkBtn()

    def enable_OkBtn(self):
        if len(self.dirname) != 0 and len(self.path) != 0 and self.rb1.isChecked() or len(self.path) != 0 and (self.rb2.isChecked()) :
            self.Ok_Btn.setEnabled(True)
        else:
            self.Ok_Btn.setDisabled(True)

# JUST FOR TEST THIS CLASS
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec_())