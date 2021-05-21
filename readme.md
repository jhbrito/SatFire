# SatFire

Guilherme Rodrigues

José Henrique Brito

Cite: Guilherme Rodrigues (2021), "WILDFIRE RISK AND BURNED AREA SIMULATOR A DEEP LEARNING APPROACH", Master thesis, School of Technlogy, Polytechnical Institute of Cavado and Ave

This repository contains
- a dataset of forest fires in Portugal in 2017 and associated metadata of land use/land cover, digital elevation model and weather
- a a prototype application to estimate buned area from land use/land cover, 

|![shape](https://user-images.githubusercontent.com/19577316/119171840-085d7700-ba5d-11eb-8e1a-69766900b72d.png)|![cos](https://user-images.githubusercontent.com/19577316/119171866-0f848500-ba5d-11eb-82bd-1822268284af.png)|![alturas](https://user-images.githubusercontent.com/19577316/119171885-16ab9300-ba5d-11eb-9ffe-a53a5d80ac6d.png)|
|:-------:|:----------:|:--------:|
|Fire Area| Land Cover | Altitude |

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
