import os
from owslib.wms import WebMapService
from PIL import Image
import numpy as np
from epsg_ident import EpsgIdent
import shapefile
from pyproj import Proj, transform
from haversine import haversine
from math import sqrt,atan,pi
import pyproj
import cv2 as cv2
import shutil
from classes_cos2018 import RGBClassesCodes
import json
import requests
from requests.auth import HTTPDigestAuth
from osgeo import gdal,osr
import sys

import matplotlib.pyplot as plt
from haversine import Unit
from PIL import ImageDraw

class GetImageMap:

    def __init__(self):

        self.gc_cos = 'COS'
        self.gc_higher = 'Higher'

    def GetWmsImage(self, size, bbox, path, map_descrp, epsg='', url='', img_format='image/png', layer='', style='', binarized=False):
        init_size = ()
        if map_descrp == self.gc_cos:
            # init_size = (2600,2600)
            init_size = (256,256)
            self.url = 'http://mapas.dgterritorio.pt/wms-inspire/cos2018v1'
            image_name = 'cos.png'
        elif map_descrp == self.gc_higher:
            init_size = (256,256)
            self.url = 'http://mapas.dgterritorio.pt/wms-inspire/mdt50m'
            image_name = 'alturas.png'
        else:
            raise Exception('Wrong Parameter "map_descrp". Only "COS" or "Higher" is allowed')

        if (url == ''):
            url = self.url

        if (epsg== ''):
            epsg='epsg:4326'

        try:
            wms = WebMapService(url, version='1.3.0')
        except:
            try:
                wms = WebMapService(url, version='1.1.1')
            except:
                raise Exception('HTTP Error: Invalide URL')

        layer_names = list(wms.contents)

        if (layer == ''):
            layer = layer_names[0]

        if (style == '' ):
            styles_names = list(wms[layer].styles)
            style = styles_names[0]

        response = wms.getmap( layers=[layer],
                            styles=[style],
                            srs=epsg,
                            bbox=bbox,
                            size=init_size,
                            format=img_format,
                            transparent=False)

        if response._response.status_code != 200:
            raise Exception('Wrong Parameters: LAYER or STYLE or EPSG or BBOX or SIZE or IMAGE FORMAT')

        data=response.read()

        out = open(image_name, 'wb')
        result = out.write(data)
        out.close()

        if binarized == True:
            img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dim = size

            #resize image
            resized = cv2.resize(img_rgb, dim, interpolation = cv2.INTER_NEAREST)
            cv2.imwrite(image_name, resized)

            bin_image = self.BuildBinaryImage(image_name, map_descrp)
            cv2.imwrite(image_name, bin_image)

        self.MoveImageToPath(path, image_name)

    def MoveImageToPath(self, path, image_name):
        image_path = path + '/' + image_name
        shutil.move(image_name, image_path)

    def get_image_numpy_format(self):
        im = Image.open(self.image_path)
        np_im = numpy.array(im)
        return np_im

    def CreateFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

    def DrawAndGetImage(self, shape, img_width, img_height, img_size, image_name, path, binarized=False):

        inProj = Proj(init='epsg:3763')
        outProj = Proj(init='epsg:4326')
        size_x = img_size[0]
        size_y = img_size[1]

        center_x = (shape.shape.bbox[0] + shape.shape.bbox[2]) / 2
        center_y = (shape.shape.bbox[1] + shape.shape.bbox[3]) / 2

        #convert to Lat/Long
        center_lat,center_lon  = transform(inProj,outProj,center_x,center_y)
        geod = pyproj.Geod(ellps='WGS84')

        #define bounding box geometry (it will be 5120mx5120m aprox.)
        width = img_width/2. # m
        height = img_height/2. # m
        rect_diag = sqrt( width**2 + height**2 )

        azimuth1 = atan(width/height)
        azimuth2 = atan(-width/height)
        azimuth3 = atan(width/height)+pi # first point + 180 degrees
        azimuth4 = atan(-width/height)+pi # second point + 180 degrees

        #calculate nem bounding box coordiates (lat/long)
        # pt4 o---------o pt1
        #     |         |
        #     |         |
        #     |         |
        # pt3 o---------o pt2

        pt1_lon, pt1_lat, _ = geod.fwd(center_lon, center_lat, azimuth1*180/pi, rect_diag)
        pt2_lon, pt2_lat, _ = geod.fwd(center_lon, center_lat, azimuth2*180/pi, rect_diag)
        pt3_lon, pt3_lat, _ = geod.fwd(center_lon, center_lat, azimuth3*180/pi, rect_diag)
        pt4_lon, pt4_lat, _ = geod.fwd(center_lon, center_lat, azimuth4*180/pi, rect_diag)

        pt1 = pt1_lat, pt1_lon
        pt2 = pt2_lat, pt2_lon
        pt3 = pt3_lat, pt3_lon
        pt4 = pt4_lat, pt4_lon

        #define new bounding box
        xmin=min(pt1[0], pt2[0], pt3[0], pt4[0])
        ymin=min(pt1[1], pt2[1], pt3[1], pt4[1])
        xmax=max(pt1[0], pt2[0], pt3[0], pt4[0])
        ymax=max(pt1[1], pt2[1], pt3[1], pt4[1])

        points_pixel=[]
        points_array=[]

        image_name = str(image_name) + '.png'
        image = Image.new('RGB', (256,256), color=(255,255,255))
        image.save(image_name)
        img = cv2.imread(image_name)

        l = shape.shape.parts

        len_l = len(l)  # how many parts of countries i.e. land and islands
        x = [i[0] for i in shape.shape.points[:]] # list of latitude
        y = [i[1] for i in shape.shape.points[:]] # list of longitude
        l.append(len(x)) # ensure the closure of the last component
        for k in range(len_l):
            # draw each component of map.
            # l[k] to l[k + 1] is the range of points that make this component
            points_part = zip(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]])
            #convert each shape point to pixel
            for points in points_part:
                point_lat,point_lon = transform(inProj,outProj,points[0],points[1])

                dist_x = haversine(pt4, (point_lat,pt4[1]))
                dist_y = haversine(pt4, (pt4[0],point_lon))

                pixel_x = int(round((size_x*dist_x)/(img_width/1000)))
                pixel_y = int(round((size_y*dist_y)/(img_height/1000)))

                points_pixel.append([pixel_x,pixel_y])

            if points_pixel: #empty parts are not considered     
                cv2.fillPoly(img, np.array([points_pixel], dtype=np.int32), color=(255,0,0))
                cv2.imwrite(image_name,img)
                points_pixel=[]

        if (binarized == True):
            #convert image to binary
            self.BinarizeImage(image_name)

        self.MoveImageToPath(path, image_name)
        

    def DeleteImage(self, image_name):
        try:
            os.remove(image_name)
        except:
            raise Exception('Impossible to delete Image: %s' + image_name)

    def BinarizeImage(self, image_name):
        #convert image to binary
        originalImage = cv2.imread(image_name)
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite(image_name, blackAndWhiteImage)

    def GetBBoxImage(self, shape, img_width, img_height, epsg=''):
        #get shape bouding box in lat/long according to image width and height
        if (epsg == ''):
            inProj = Proj(init='epsg:3763')
        else:
            inProj = Proj(init=epsg)

        outProj = Proj(init='epsg:4326')

        center_x = (shape.shape.bbox[0] + shape.shape.bbox[2]) / 2
        center_y = (shape.shape.bbox[1] + shape.shape.bbox[3]) / 2

        #convert to Lat/Long (inProj coordinates)
        center_lat,center_lon  = transform(inProj,outProj,center_x,center_y)
        geod = pyproj.Geod(ellps='WGS84')

        #define bounding box geometry (it will be 5120mx5120m aprox.)
        width = img_width/2 # m
        height = img_height/2 # m
        rect_diag = sqrt( width**2 + height**2 )

        azimuth1 = atan(width/height)
        azimuth2 = atan(-width/height)
        azimuth3 = atan(width/height)+pi # first point + 180 degrees
        azimuth4 = atan(-width/height)+pi # second point + 180 degrees

        #calculate nem bounding box coordiates (lat/long)
        # pt4 o---------o pt1
        #     |         |
        #     |         |
        #     |         |
        # pt3 o---------o pt2

        pt1_lon, pt1_lat, _ = geod.fwd(center_lon, center_lat, azimuth1*180/pi, rect_diag)
        pt2_lon, pt2_lat, _ = geod.fwd(center_lon, center_lat, azimuth2*180/pi, rect_diag)
        pt3_lon, pt3_lat, _ = geod.fwd(center_lon, center_lat, azimuth3*180/pi, rect_diag)
        pt4_lon, pt4_lat, _ = geod.fwd(center_lon, center_lat, azimuth4*180/pi, rect_diag)

        pt1 = pt1_lat, pt1_lon
        pt2 = pt2_lat, pt2_lon
        pt3 = pt3_lat, pt3_lon
        pt4 = pt4_lat, pt4_lon

        #check distances bbox points(this is not necessary for code)
        x_dist = haversine(pt1, pt2)
        y_dist = haversine(pt3, pt4)
        z_dist = haversine(pt3, pt2)
        w_dist = haversine(pt1, pt4)
        r_dist = haversine(pt1, pt3)
        q_dist = haversine(pt2, pt4)

        #define new bounding box
        bbox = [min(pt1[0], pt2[0], pt3[0], pt4[0]), min(pt1[1], pt2[1], pt3[1], pt4[1]), max(pt1[0], pt2[0], pt3[0], pt4[0]), max(pt1[1], pt2[1], pt3[1], pt4[1])]

        return bbox

    def BuildBinaryImage(self, img_name, map_descrp):

        codes = RGBClassesCodes()
        cos_list = ['2',('3','4'),'5','6']
        img = cv2.imread(img_name)
        rgb_list={}

        # grab the image dimensions
        h = img.shape[0]
        w = img.shape[1]

        bin_image = np.ones([256,256], dtype=np.uint8)*255

        # build new image, pixel by pixel
        if map_descrp == self.gc_cos: #build cos map image
            rgb_list = codes.BuildDynamicDict(cos_list)
            for y in range(0, h):
                for x in range(0, w):
                    if tuple(img[x,y]) in rgb_list.keys():
                        bin_image[x, y] = rgb_list[tuple(img[x,y])]

        elif map_descrp == self.gc_higher: #build higher map image
            rgb_list = codes.rgb_highercode_dict
            for y in range(0, h):
                for x in range(0, w):
                    if tuple(img[x,y]) in rgb_list.keys():
                        bin_image[x, y] = rgb_list[tuple(img[x,y])]

        return bin_image

    def BuildJsonFile(self, shape):
        data={}

        inProj = Proj(init='epsg:3763')
        outProj = Proj(init='epsg:4326')

        center_x = (shape.shape.bbox[0] + shape.shape.bbox[2]) / 2
        center_y = (shape.shape.bbox[1] + shape.shape.bbox[3]) / 2

        #convert to Lat/Long
        center_lat,center_lon  = transform(inProj,outProj,center_x,center_y)

        data = self.GetWeatherConditions(shape,center_lat,center_lon)

        data.update({'area': shape.record.Area_SIG})

        return data

    def GetWeatherConditions(self, shape, lat, longt):

        lat_long = str(lat)[:6]+' '+str(longt)[:6]
        url = 'https://api.worldweatheronline.com/premium/v1/past-weather.ashx?'+'date='+str(shape.record.DHInicio[:10])+'&includelocation=yes'+'&tp=1'+'&key=bf3a4f7d854f4be5bb8174222202706'+'&q='+lat_long+'&format=json'

        # get aproximately hour (entire number)
        try:
            if int(shape.record.DHInicio[14:16]) >= 30:
                hour = int(shape.record.DHInicio[11:13]) + 1
            else:
                hour = int(shape.record.DHInicio[11:13])

            if hour == 24:
                hour = 0
        except:
            raise Exception('HTTP Error: Invalide URL')


        myResponse = requests.get(url)

        if(myResponse.ok):
            json_data = json.loads(myResponse.content)

            hourly=json_data['data']['weather'][0]['hourly'][hour]

            out_dict={
                'date': shape.record.DHInicio[:10],
                'hour':hour,
                'humidity': hourly['humidity'],
                'tempC': hourly['tempC'],
                'windspeedKmph': hourly['windspeedKmph'],
                'winddir16Point': hourly['winddir16Point'],
                'winddirDegree': hourly['winddirDegree'],
                'precipMM': hourly['precipMM'],
                'cloudcover': hourly['cloudcover'],
                'WindGustKmph': hourly['WindGustKmph'],
                'lat/long': lat_long,
                'area': shape.record.Area_SIG
                }
        else:
        # If response code is not ok (200), print the resulting http error code with description
            # myResponse.raise_for_status()
            out_dic={}

        return out_dict

    def GetHighersImage(self, bbox, path):

        # this allows GDAL to go throw Python Exceptions
        gdal.UseExceptions()

        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'NO')  # we don't want free space to be an issue here

        InputImage = r'E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Alturas_Portugal_completo.tif'

        try:
            ds = gdal.Open(InputImage, gdal.GA_ReadOnly)
        except RuntimeError as e:
            print('Unable to open INPUT.tif')
            print(e)
            sys.exit(1)

        projection = '-projwin' + ' ' + str(bbox[0]) + ' ' + str(bbox[3]) + ' ' + str(bbox[2]) + ' ' + str(bbox[1])

        # More options in: http://manpages.ubuntu.com/manpages/xenial/man1/gdal_translate.1.html
        options_list = [
            '-ot Byte',
            '-of PNG',
            projection,         #-projwin -8.289023852308949 37.63633719367372 -7.485264582241154 37.05456857914846',
            '-outsize 256 256',
            '-scale -3.30125 2369.7'
        ]
        options_string = " ".join(options_list)
        filepath = path + '/higher.png'
        gdal.Translate(filepath, ds, options=options_string)
        ds = None
        new_ds = None

# JUST FOR TEST THIS CLASS
# if __name__ == "__main__":
    # image = GetImageMap()
    # image.CreateFolder(<path_to_create_folder>)
    # #draw de shape and save respective image (256x256)
    # shape_image = image.DrawAndGetImage(<shape(iterated)>, 5120, 5120, (256,256), 'shape', <new_path_with_created_folder>, binarized=True)

    # #get and save images from WMS services
    # bounding_box = image.GetBBoxImage(<shape(iterated)>, 5120, 5120, <shape_epsg>)

    # wms_image_cos = image.GetWmsImage((256,256), <bounding_box>, <new_path_with_created_folder>, 'COS')
    # wms_image_higher = image.GetWmsImage((256,256), <bounding_box>, <new_path_with_created_folder>, 'Higher')

    # # build json file
    # json_data[min_limit] = image.BuildJsonFile(<shape(iterated)>)