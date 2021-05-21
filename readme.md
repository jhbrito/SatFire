# SatFire

Guilherme Rodrigues

José Henrique Brito

## READ ME PLEASE:

This are the very first instructions you must follow before execute any code:

### 1. Please verify you have the follow libraries installed on your machine, if not please install it as follows:

***NOTE1: If the follows commands give you error of connection, you must connect to a VPN to execute them.***

***NOTE2: All code presented on this repository was built under Python 3.7.4 version on Windows Platform in Visual Studio Code.***

Commands:

python -m pip install --upgrade pip

python -m pip install -U pylint --user 							"This step is optional

python -m pip install -U OWSLib --user

python -m pip install -U Pillow --user

python -m pip install -U pandas --user

python -m pip install -U pyshp --user 							"to open shapefiles

python -m pip install -U matplotlib --user

python -m pip install -U opencv-python --user

pylint --generate-rcfile > .pylintrc             					"This step is optional - In case your interpreter be pylint and it does not recognize cv2 members.

python -m pip install -U pyproj --user  						"This library is for geographic coordinates conversion

python -m pip install -U epsg-ident --user

python -m pip install -U Shapely-1.6.4.post2-cp37-cp37m-win_amd64.whl --user 		"Probably this lib is not necessary

python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose 	"Probably this lib is not necessary

python -m pip install -U PySide2 --user "This is to install QT for Python to use in the interface

pip install tensorflow==2.0.0 " tensorflow 2.1.0 will not work properly on cpu version

pip install Keras

pip install -U scikit-learn


### 2. Possible Errors may occure:


1. Problem: "SystemError: error return without exception set" 

    Solution: Set environmental variable ***PYDEVD_USE_FRAME_EVAL=NO*** in Pycharm's run configurations menu.
