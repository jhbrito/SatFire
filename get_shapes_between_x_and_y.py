from pyproj import Proj, transform
from haversine import haversine
from epsg_ident import EpsgIdent
import shutil
import shapefile 
from get_image_map import GetImageMap
import os

class ProcesseShapes:

    def __init__(self, path, len_min=0, len_max=0, dirname_path=''):
        self.min = len_min
        self.max = len_max
        self.path = path
        self.dirname_path = dirname_path

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
                if ((x_dist <= self.max and x_dist >= self.min) or (y_dist <= self.max and y_dist >= self.min) ):
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

    def process_file(self, url):
        sf = shapefile.Reader(self.path)
        total = 0
        print("Processing new shape file...")
        image = GetImageMap()

        #create folder to save the information about the shapefile
        head_tail = os.path.split(self.path)
        path = head_tail[0] + '/imagem'
        image.CreateFolder(path)  

        for shape in sf.iterShapeRecords(): #loop shapefile
            total = total + 1

            #create folder to save the information about the shapefile
            new_path = path + '/' + str(total)
            image.CreateFolder(new_path) 

            #draw de shape and save respective image (256x256)
            shape_image = image.DrawAndGetImage(shape, 5120, 5120, (256,256), total, new_path, binarized=True)

            #get and save images from WMS services
            bounding_box = image.GetBBoxImage(shape, 5120, 5120, self.epsg), 
            
            wms_image = image.GetWmsImage((256,256), bounding_box, new_path)

            # image_array = image.get_image_numpy_format()
            if total == 2:
                break
        print("File Processing is finished!")

    def read_epsg(self):
        ident = EpsgIdent()
        prj_file = self.path.replace('.shp', '.prj')
        ident.read_prj_from_file(prj_file)
        epsg = ident.get_epsg()
        return ('EPSG:' + str(epsg))

   
