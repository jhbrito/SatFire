# READ ME PLEASE:

This are the very first instructions you must follow before execute any code:

## 1. Please verify you have the follow libraries installed on you machine, if not please install it as follows:

***NOTE: If the follows commands give you error of connection, you must connect to a VPN to execute them.***

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