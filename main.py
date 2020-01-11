import sys
from PySide2 import QtCore, QtWidgets 
from Display_ShapeFile import DisplayAllShapes
from get_shapes_between_x_and_y import DeleteExtraShapes
from get_image_map import GetImageMap


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
        self.Exit_Btn = QtWidgets.QPushButton(self.centralWidget)

        self.file_Select_Btn.setObjectName("file_Select_Btn")
        self.file_Select_Btn.setText("Load Shape File")
        self.Ok_Btn.setObjectName("Ok_Btn")
        self.Ok_Btn.setText("OK")
        self.Ok_Btn.setDisabled(True)
        self.Exit_Btn.setObjectName("Exit_Btn")
        self.Exit_Btn.setText("Exit")

        self.gridLayout.addWidget(self.file_Input_Label,1,1)
        self.gridLayout.addWidget(self.file_Select_Btn,1,2)
        self.gridLayout.addWidget(self.Ok_Btn,2,1)
        self.gridLayout.addWidget(self.Exit_Btn,2,2)

        MainWindow.setCentralWidget(self.centralWidget)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        Ui_MainWindow.__init__(self)
        QtWidgets.QMainWindow.__init__(self)
        self.path = ''

        # Initialize UI
        self.setupUi(self)
        self.file_Select_Btn.clicked.connect(self.getFilePath)
        self.Exit_Btn.clicked.connect(QtCore.QCoreApplication.instance().quit)   #close application
        self.Ok_Btn.clicked.connect(self.clickOkBtn)

    def tr(self, text):
        return QtCore.QObject.tr(self, text)

    def getFilePath(self):
        self.path, ok = QtWidgets.QFileDialog.getOpenFileName(self, self.tr("Load File"), self.tr("~/Desktop/"), self.tr("Shape Files (*.shp)"))
        self.file_Input_Label.setText(self.path)

        if ok:
            self.Ok_Btn.setEnabled(True)
        else:
            self.Ok_Btn.setDisabled(True)

    def clickOkBtn(self):

        # 1.DISPLAY ALL SHAPES
        # file1 = DisplayAllShapes(self.path)
        # file1.open_all_shapes_png()
        
        # 2.CREATE NEW SHAPEFILE WITH SHAPES ONLY BETWEEN X AND Y METERS OF DISTANCE
        # file2 = DeleteExtraShapes(40, 5120, self.path)
        # file2.delete_shapes_out_of_range()

        # 3.GET MAP IMAGE FROM WMS SERVICE AND RETURN THE IMAGE AS A NUMPY ARRAY
        image = GetImageMap('http://mapas.dgterritorio.pt/wms/hrl-PTContinente', 256, (-11.0795, 36.3292, -4.68523, 42.7511))
        image.get_image()
        image_array = image.save_image('E:/OneDrive - Instituto Politécnico do Cávado e do Ave/Desktop_backup/Tese/dados_tese','teste_imagem')
        print (image_array)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())