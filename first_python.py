from owslib.wms import WebMapService

wms = WebMapService('http://mapas.dgterritorio.pt/wms-inspire/cos2015v1', version='1.3.0')
print(wms['COS2015v1.0'].boundingBox)
print(wms['COS2015v1.0'].boundingBoxWGS84)
# GetMap (image/jpeg)
response = wms.getmap( layers=['COS2015v1.0'],
                    styles=['default'],
                    srs='EPSG:4326',
                    bbox=(-10.1905, 36.7643, -5.71298, 42.1896),
                    size=(400, 400),
                    format='image/png',)

out = open('COS2015.jpeg', 'wb')
out.write(response.read())
out.close()

# GetFeatureInfo (text/html)
# response = wms.getfeatureinfo(
#     layers=['bvv:gmd_ex'],
#     srs='EPSG:31468',
#     bbox=(4500000,5500000,4505000,5505000),
#     size=(500,500),
#     format='image/jpeg',
#     query_layers=['bvv:gmd_ex'],
#     info_format="text/html",
#     xy=(250,250))

# out = open('getfeatureinfo-response.html', 'wb')
# out.write(response.read())
# out.close()