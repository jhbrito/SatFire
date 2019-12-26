from owslib.wms import WebMapService

wms = WebMapService('http://mapas.dgterritorio.pt/wms/hrl-PTContinente', version='1.3.0')

# GetMap (image/jpeg)
response = wms.getmap( layers=['HRL_Grau_Coberto_Florestal_2015'],
                    styles=['default'],
                    srs='EPSG:4326',
                    bbox=(-11.0795, 36.3292, -4.68523, 42.7511),
                    size=(2600, 2600),
                    format='image/png',)

out = open('floresta_2015.jpeg', 'wb')
out.write(response.read())
out.close()

