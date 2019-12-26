# from PIL import Image
from io import StringIO
# import requests
from owslib.wms import WebMapService
import matplotlib.pyplot as plt
# import cartopy


wms_map = WebMapService('http://mapas.dgterritorio.pt/wms-inspire/cos2015v1', version='1.3.0')

type = wms_map.identification.type
print(wms_map.identification.type)
print(wms_map.identification.title)
print(list(wms_map.contents))
print(wms_map['COS2015v1.0'].title)
print(wms_map['COS2015v1.0'].queryable)
print(wms_map['COS2015v1.0'].opaque)
print(wms_map['COS2015v1.0'].boundingBox)
print(wms_map['COS2015v1.0'].boundingBoxWGS84)
print(wms_map['COS2015v1.0'].crsOptions)
print(wms_map['COS2015v1.0'].styles)
print([op.name for op in wms_map.operations])
print(wms_map.getOperationByName('GetMap').methods)
print(wms_map.getOperationByName('GetMap').formatOptions)

img = wms_map.getmap(   layers=['COS2015v1.0'],
                    styles=['default'],
                    srs='EPSG:4326',
                    bbox=(-112, 36, -106, 41),
                    size=(300, 250),
                    format='image/png',
                    transparent=True)
print('IMAGEM::\n\n')
out = open('jpl_mosaic_visb.png', 'wb')
result = out.write(img.read())
out.close()