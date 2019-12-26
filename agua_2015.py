from owslib.wms import WebMapService

wms = WebMapService('http://mapas.dgterritorio.pt/wms/hrl-PTContinente', version='1.3.0')
# GetMap (image/jpeg)
# type = wms.identification.type
# print(wms.identification.type)
# print(wms.identification.title)
# print(list(wms.contents))
# print(wms['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].title)
# print(wms['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].queryable)
# print(wms['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].opaque)
# print(wms['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].boundingBox)
# print(wms['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].boundingBoxWGS84)
# print(wms['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].crsOptions)
# print(wms['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].styles)
# print(wms.getOperationByName('GetMap').methods)
# print(wms.getOperationByName('GetMap').formatOptions)
response = wms.getmap( layers=['HRL_Zonas_Humidas_e_Corpos_Agua_2015'],
                    styles=['default'],
                    srs='EPSG:4326',
                    bbox=(-11.0795, 36.3292, -4.68523, 42.7511),
                    size=(2600, 2600),
                    format='image/png',)

out = open('agua_2015.jpeg', 'wb')
out.write(response.read())
out.close()
