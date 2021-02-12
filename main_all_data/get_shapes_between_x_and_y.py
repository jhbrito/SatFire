from pyproj import Proj, transform
from haversine import haversine
from epsg_ident import EpsgIdent
import shutil
import shapefile
from get_image_map import GetImageMap
import os
import json
   
import matplotlib.pyplot as plt
import cv2
from haversine import Unit
from PIL import Image, ImageDraw


class ProcesseShapes:

    def __init__(self, path, len_min=0, len_max=0, dirname_path=''):
        self.min = len_min
        self.max = len_max
        self.path = path
        self.dirname_path = dirname_path
        self.value = 3 #(km)

        self.epsg = self.read_epsg() #EPSG coordinates from uploaded shapefile
        self.inProj = Proj(init=self.epsg)
        self.outProj = Proj(init='epsg:4326') #lat long coordinates

    def delete_shapes_out_of_range(self):

        # sf = shapefile.Reader(r"E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")
        sf = shapefile.Reader(self.path)

        print("Builting the new shape file ")

        # new_doc = 'E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\copy'
        new_doc = self.dirname_path
        new_shapefile = shapefile.Writer(new_doc)
        new_shapefile.fields = sf.fields[1:] # skip first deletion field

        for shape in sf.iterShapeRecords(): #loop shapefile

            xmin = shape.shape.bbox[0]
            xmax = shape.shape.bbox[2]
            ymin = shape.shape.bbox[1]
            ymax = shape.shape.bbox[3]

            x_minpoint = transform(self.inProj,self.outProj,xmin,ymin)
            x_maxpoint = transform(self.inProj,self.outProj,xmax, ymin)
            y_minpoint = transform(self.inProj,self.outProj,xmin,ymin)
            y_maxpoint = transform(self.inProj,self.outProj,xmin,ymax)

            x_dist = haversine(x_minpoint, x_maxpoint)
            y_dist = haversine(y_minpoint, y_maxpoint)

            if x_dist != "" and y_dist != "":
                if ((x_dist <= self.max and x_dist >= self.min) or (y_dist <= self.max and y_dist >= self.min)) and shape.record.HoraAlerta != '':
                    new_shapefile.record(*shape.record)
                    new_shapefile.shape(shape.shape)
                else:
                    continue
            else:
                continue
            del x_dist, y_dist, xmin, xmax, ymin, ymax, x_minpoint, y_minpoint, x_maxpoint, y_maxpoint
        new_shapefile.close()

        # create the PRJ file
        prj_file_old = self.path.replace('.shp', '.prj') #old proj file
        prj_file_new = new_doc.replace('.shp', '.prj') #new proj file
        shutil.copy(prj_file_old, prj_file_new) #copy the old proj file for new path
        print("->File Finished!<-")

    def process_file(self):
        sf = shapefile.Reader(self.path)

        print("Processing new shape file...")
        image = GetImageMap()

        #create folder to save the information about the shapefile
        head_tail = os.path.split(self.path)
        path = head_tail[0] + '/dataset_review'
        image.CreateFolder(path)

        min_limit = self.GetNumberRequestsMade(path)
        max_limit = min_limit + 1000

        json_data = {}
        idx = 0
        for shape in sf.iterShapeRecords(): #loop shapefile

            if idx >= min_limit and idx < max_limit:
                min_limit = min_limit + 1

                #create folder to save the information about the shapefile
                new_path = path + '/' + str(min_limit)
                image.CreateFolder(new_path)

                #draw de shape and save respective image (256x256)
                # self.ShowRealImage(shape.shape)
                shape_image = image.DrawAndGetImage(shape, 1000, 1000, (256,256), 'shape', new_path, binarized=True)

                #get and save images from WMS services
                bounding_box = image.GetBBoxImage(shape, 1000, 1000, self.epsg)

                wms_image_cos = image.GetWmsImage((256,256), bounding_box, new_path, 'COS', binarized=False)
                wms_image_higher = image.GetWmsImage((256,256), bounding_box, new_path, 'Higher', binarized=False)
                image.GetHighersImage(bounding_box, new_path)

                # build json file
                # json_data[min_limit] = image.BuildJsonFile(shape)

                # insert into txt file
                # self.CalculateShapeWidth(shape, total, file1, file2)
                # image_array = image.get_image_numpy_format()

            if idx >= max_limit:
                break

            idx = idx + 1

        # self.UpdateJsonFile(json_data, path)

        print("File Processing is finished!")

    def read_epsg(self):
        ident = EpsgIdent()
        prj_file = self.path.replace('.shp', '.prj')
        ident.read_prj_from_file(prj_file)
        epsg = ident.get_epsg()
        return ('EPSG:' + str(epsg))

    def CalculateShapeWidth(self, shape, number, file1, file2):
        xmin = shape.shape.bbox[0]
        xmax = shape.shape.bbox[2]
        ymin = shape.shape.bbox[1]
        ymax = shape.shape.bbox[3]

        x_minpoint = transform(self.inProj,self.outProj,xmin,ymin)
        x_maxpoint = transform(self.inProj,self.outProj,xmax, ymin)
        y_minpoint = transform(self.inProj,self.outProj,xmin,ymin)
        y_maxpoint = transform(self.inProj,self.outProj,xmin,ymax)

        x_dist = haversine(x_minpoint, x_maxpoint)
        y_dist = haversine(y_minpoint, y_maxpoint)

        if x_dist != "" and y_dist != "":
            if (x_dist >= self.value or y_dist >= self.value):
                file1.write("%i \n" %number)
            else:
                file2.write("%i \n" %number)

        del x_dist, y_dist, xmin, xmax, ymin, ymax, x_minpoint, y_minpoint, x_maxpoint, y_maxpoint

    def GetNumberRequestsMade(self, directory):
        all_subdirs = os.listdir(directory)
        if all_subdirs:
            total = int(len(all_subdirs)-1)
        else:
            total = 0

        return total

    def UpdateJsonFile(self, data, directory):
        path = directory + '/data.json'
        new_data={}

        if not os.path.exists(path):
            with open(path, "w") as jsonFile:
                json.dump(data, jsonFile)
        else:
            with open(path, "r") as jsonFile:
                old_data = json.load(jsonFile)
            new_data = old_data
            new_data.update(data)
            with open(path, "w") as jsonFile:
                json.dump(new_data, jsonFile)

    def ShowRealImage(self, shape):

        l = shape.parts
    
        len_l = len(l)  # how many parts of countries i.e. land and islands
        x = [i[0] for i in shape.points[:]] # list of latitude
        y = [i[1] for i in shape.points[:]] # list of longitude
        l.append(len(x)) # ensure the closure of the last component
        for k in range(len_l):
            # draw each component of map.
            # l[k] to l[k + 1] is the range of points that make this component
            plt.plot(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'k-', )
            plt.fill(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]])
        plt.show()

# JUST FOR TEST THIS CLASS
# if __name__ == "__main__":
    # a = 1
    # if a == 1:
    #     file2 = ProcesseShapes(<path_to_shapefile>,<shape_width_min_value>, <shape_width_max_value>, <path_to_save_new_shapefile>)
    #     file2.delete_shapes_out_of_range()
    # else:
    #     file2 = ProcesseShapes(<path_to_shapefile>)
    #     file2.process_file()