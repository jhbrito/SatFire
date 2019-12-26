from owslib.wms import WebMapService

# wms = WebMapService('http://www.igeo.pt/WMS/Natureza/AreasArdidas', version='1.3.0')
wms = WebMapService('http://si.icnf.pt/wms/ardida_2017?service=wms&version=1.1.1&request=GetCapabilities', version='1.1.1')
# type = wms.identification.type
# print(wms.identification.type)
# print(wms.identification.title)
# print(list(wms.contents))
# print(wms['ardida_2017'].title)
# print(wms['ardida_2017'].queryable)
# print(wms['ardida_2017'].opaque)
# print(wms['ardida_2017'].boundingBox)
# print(wms['ardida_2017'].boundingBoxWGS84)
# print(wms['ardida_2017'].crsOptions)
# print(wms['ardida_2017'].styles)
# print(wms.getOperationByName('GetMap').methods)
print(wms.getOperationByName('GetMap').formatOptions)
response = wms.getmap( layers=['ardida_2017'],
                    styles=['BDG:ardida_2017'],
                    # srs='EPSG:3763',
                    srs='EPSG:4326',
                    # bbox=(-111304.9921875, -291967.625, 160375.421875, 273254.84375),
                    bbox=(-11.0795, 36.3292, -4.68523, 42.7511),
                    size=(2600, 2600),
                    format='image/png',
                    transparent=False)

out = open('areas_ardidas_layer.png', 'wb')
out.write(response.read())
out.close()

