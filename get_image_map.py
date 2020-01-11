from owslib.wms import WebMapService
from PIL import Image
import numpy

class GetImageMap:

    def __init__(self, url_wms, size, bbox):
        self.url = url_wms
        self.size = size
        self.bbox = bbox

    def get_image(self):

        # wms = WebMapService('http://mapas.dgterritorio.pt/wms/hrl-PTContinente', version='1.3.0')
        wms = WebMapService(self.url, version='1.3.0')

        # GetMap (image/jpeg)
        self.response = wms.getmap( layers=['HRL_Grau_Coberto_Florestal_2015'],
                            styles=['default'],
                            srs='EPSG:4326',
                            bbox = self.bbox,
                            size=(self.size, self.size),
                            format='image/png',)
        
    def save_image(self, path, image_name):
        image_path = path + '/' + image_name + '.png'
        out = open(image_path, 'wb')
        out.write(self.response.read())
        out.close()

        im = Image.open(image_path)
        np_im = numpy.array(im)
        return np_im
