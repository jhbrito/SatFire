import os
from owslib.wms import WebMapService
from PIL import Image
import numpy

class GetImageMap:

    def __init__(self, url_wms, size, bbox, epsg):
        self.url = url_wms
        self.size = size
        self.bbox = bbox
        self.epsg = epsg

    def get_image(self):

        # wms = WebMapService('http://mapas.dgterritorio.pt/wms/hrl-PTContinente', version='1.3.0')
        wms = WebMapService(self.url, version='1.3.0')

        self.response = wms.getmap( layers=['ardida_2017'],
                            styles=['BDG:ardida_2017'],
                            # srs='EPSG:3763',
                            srs=self.epsg,
                            bbox=self.bbox,
                            size=(self.size, self.size),
                            format='image/png',
                            transparent=False)

# GetMap (image/jpeg)
        # self.response = wms.getmap( layers=['HRL_Grau_Coberto_Florestal_2015'],
        #                     styles=['default'],
        #                     srs='EPSG:3763',
        #                     bbox=self.bbox,
        #                     size=(self.size, self.size),
        #                     format='image/png',)

    def save_image(self, path, image_name):
        self.image_path = path + '/' + str(image_name) + '.png'
        out = open(self.image_path, 'wb')
        out.write(self.response.read())
        out.close()

    def get_image_numpy_format(self):
        im = Image.open(self.image_path)
        np_im = numpy.array(im)
        return np_im
    
    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)


       