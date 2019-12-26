from owslib.wms import WebMapService

wms = WebMapService('http://mapas.dgterritorio.pt/wms-inspire/cos2015v1', version='1.3.0')
# GetMap (image/jpeg)
type = wms.identification.type
print(wms.identification.type)
print(wms.identification.title)
print(list(wms.contents))
print(wms['COS2015v1.0'].boundingBox)
print(wms['COS2015v1.0'].boundingBoxWGS84)
print(wms['COS2015v1.0'].crsOptions)
print(wms['COS2015v1.0'].styles)
print(wms.getOperationByName('GetMap').methods)
print(wms.getOperationByName('GetMap').formatOptions)
response = wms.getmap( layers=['COS2015v1.0'],
                    styles=['default'],
                    srs='EPSG:4326',
                    bbox=(-10.1905, 36.7643, -5.71298, 42.1896),
                    size=(400, 600),
                    format='image/png',)

out = open('COS2015.jpeg', 'wb')
out.write(response.read())
out.close()
