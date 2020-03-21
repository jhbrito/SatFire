import sys
from PySide2 import QtCore, QtWidgets
from Display_ShapeFile import DisplayAllShapes
from get_shapes_between_x_and_y import ProcesseShapes
from get_image_map import GetImageMap


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
        self.horizontalLayout5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout7 = QtWidgets.QHBoxLayout()
        self.verticalLayout2 = QtWidgets.QVBoxLayout()
        self.verticalLayout3 = QtWidgets.QVBoxLayout()


        self.file_Select_Btn = QtWidgets.QPushButton()
        self.file_Input_Label = QtWidgets.QLineEdit()
        self.file_save_path_Btn = QtWidgets.QPushButton()
        self.file_save_path = QtWidgets.QLineEdit()
        self.rb1 = QtWidgets.QRadioButton('Processing Shapefile:')
        self.rb2 = QtWidgets.QRadioButton('Generate Images:')
        self.input_max = QtWidgets.QLineEdit('5.120')
        self.input_min = QtWidgets.QLineEdit('0.040')
        self.label_max = QtWidgets.QLabel('Max Width (Km):')
        self.label_min = QtWidgets.QLabel('Min Width (Km):')
        self.Ok_Btn = QtWidgets.QPushButton()
        self.Exit_Btn = QtWidgets.QPushButton()

        self.file_Select_Btn.setObjectName("file_Select_Btn")
        self.file_Select_Btn.setText("Load Shape File")
        self.file_save_path_Btn.setText("Select Save Path")
        self.Ok_Btn.setObjectName("Ok_Btn")
        self.Ok_Btn.setText("OK")
        self.Ok_Btn.setDisabled(True)
        self.label_min.setVisible(False)
        self.label_max.setVisible(False)
        self.input_min.setVisible(False)
        self.input_max.setVisible(False)
        self.file_save_path.setVisible(False)
        self.file_save_path_Btn.setVisible(False)
        self.input_max.setMaximumWidth(300)
        self.input_min.setMaximumWidth(300)
        self.Exit_Btn.setObjectName("Exit_Btn")
        self.Exit_Btn.setText("Exit")


        self.horizontalLayout1.addWidget(self.file_Input_Label)
        self.horizontalLayout1.addWidget(self.file_Select_Btn)
        self.horizontalLayout7.addWidget(self.file_save_path)
        self.horizontalLayout7.addWidget(self.file_save_path_Btn)        
        self.horizontalLayout2.addWidget(self.rb1, 50, QtCore.Qt.AlignLeft)
        self.horizontalLayout3.addWidget(self.label_min, 0, QtCore.Qt.AlignLeft)
        self.horizontalLayout3.addWidget(self.input_min, 0, QtCore.Qt.AlignLeft)
        self.horizontalLayout3.addWidget(self.label_max, 0, QtCore.Qt.AlignLeft)
        self.horizontalLayout3.addWidget(self.input_max, 0, QtCore.Qt.AlignLeft)
        self.horizontalLayout4.addWidget(self.rb2, 50, QtCore.Qt.AlignLeft)
        self.horizontalLayout6.addWidget(self.Ok_Btn)
        self.horizontalLayout6.addWidget(self.Exit_Btn)

        self.verticalLayout.addLayout(self.horizontalLayout1)
        self.verticalLayout.addLayout(self.horizontalLayout7)

        self.verticalLayout2.addLayout(self.horizontalLayout2)
        self.verticalLayout2.addLayout(self.horizontalLayout4)
        self.verticalLayout3.addLayout(self.horizontalLayout3)

        self.horizontalLayout5.addLayout(self.verticalLayout2)
        self.horizontalLayout5.addLayout(self.verticalLayout3)

        self.verticalLayout.addLayout(self.horizontalLayout5)
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
        self.file_save_path_Btn.clicked.connect(self.getSavePath)
        self.Exit_Btn.clicked.connect(QtCore.QCoreApplication.instance().quit)   #close application
        self.Ok_Btn.clicked.connect(self.clickOkBtn)
        self.rb2.setChecked(True)
        self.rb1.toggled.connect(lambda: self.rb_clicked())
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
        # 1.DISPLAY ALL SHAPES
        # file1 = DisplayAllShapes(self.path)
        # file1.open_all_shapes_png()
        
        # 2.CREATE NEW SHAPEFILE WITH SHAPES ONLY BETWEEN X AND Y METERS OF DISTANCE
        if self.rb1.isChecked():
            file2 = ProcesseShapes(self.path, float(self.input_min.text()), float(self.input_max.text()), self.dirname)
            file2.delete_shapes_out_of_range()

        elif self.rb2.isChecked():
            file2 = ProcesseShapes(self.path)
            file2.process_file('http://si.icnf.pt/wms/ardida_2017?service=wms&version=1.3.0&request=GetCapabilities')

        # 3.GET MAP IMAGE FROM WMS SERVICE AND RETURN THE IMAGE AS A NUMPY ARRAY
        # image = GetImageMap('http://mapas.dgterritorio.pt/wms/hrl-PTContinente', 256, (-11.0795, 36.3292, -4.68523, 42.7511))
        # image.get_image()
        # image_array = image.save_image('E:/OneDrive - Instituto Politécnico do Cávado e do Ave/Desktop_backup/Tese/dados_tese','teste_imagem')
        # print (image_array)
        self.Ok_Btn.setEnabled(True)
    def rb_clicked(self):
        #on radio buttons click, the interface layout changes
        if self.rb1.isChecked():
            self.input_max.setVisible(True)
            self.input_min.setVisible(True)
            self.label_min.setVisible(True)
            self.label_max.setVisible(True)
            self.file_save_path.setVisible(True)
            self.file_save_path_Btn.setVisible(True)
        elif self.rb2.isChecked():
            self.input_max.setVisible(False)
            self.input_min.setVisible(False)
            self.label_min.setVisible(False)
            self.label_max.setVisible(False)
            self.file_save_path.setVisible(False)
            self.file_save_path_Btn.setVisible(False)

        self.enable_OkBtn()

    def enable_OkBtn(self):
        if len(self.dirname) != 0 and len(self.path) != 0 and self.rb1.isChecked() or len(self.path) != 0 and (self.rb2.isChecked()) :
            self.Ok_Btn.setEnabled(True)
        else:
            self.Ok_Btn.setDisabled(True)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    