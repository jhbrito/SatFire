from pyproj import Proj, transform 
from haversine import haversine
from epsg_ident import EpsgIdent
from math import sqrt,atan,pi
import pyproj
import shutil
import shapefile 
from get_image_map import GetImageMap



class DeleteExtraShapes:

    def __init__(self, path, len_min=0, len_max=0, dirname_path=''):
        self.min = len_min
        self.max = len_max
        self.path = path
        self.dirname_path = dirname_path

        self.inProj = Proj(init='epsg:3763')
        self.outProj = Proj(init='epsg:4326')

        self.url_water = 'https://inspire.apambiente.pt/getogc/services/INSPIRE/AM_WaterBodyForWFD_WFDRiver/MapServer/WMSServer?SERVICE=WMS&REQUEST=GetCapabilities'
        self.url_vegetation = 'http://mapas.dgterritorio.pt/wms/hrl-PTContinente'

        self.epsg = self.read_epsg()

    def delete_shapes_out_of_range(self):
               
        # sf = shapefile.Reader(r"E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")
        sf = shapefile.Reader(self.path)
        
        print("Builting the new shape file ")
        # new_doc = 'E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\copy'
        new_doc = self.dirname_path
        w = shapefile.Writer(new_doc)
        w.fields = sf.fields[1:] # skip first deletion field

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
                    w.record(*shape.record)
                    w.shape(shape.shape)
                else:
                    continue
            else:
                continue
            del x_dist, y_dist, xmin, xmax, ymin, ymax, x_minpoint, y_minpoint, x_maxpoint, y_maxpoint
        w.close()

        # create the PRJ file
        prj_file_old = self.path.replace('.shp', '.prj')
        new_doc.replace('.shp', '')
        prj_file_new = new_doc + '.prj'
        shutil.copy(prj_file_old, prj_file_new)

        print("->File Finished!<-")

    def process_file(self, url):
        sf = shapefile.Reader(self.path)
        total = 0
        print("Processing new shape file...")
        
        for shape in sf.iterShapeRecords(): #loop shapefile
            total = total + 1
            path = 'E:/OneDrive - Instituto Politécnico do Cávado e do Ave/Desktop_backup/Tese/dados_tese' + '/imagem'
            image = GetImageMap(url, 256, shape.shape.bbox, self.epsg)
            image.get_image()
            image.createFolder(path)
            image.save_image(path, total)
            image_array = image.get_image_numpy_format()
            
        print("File Processing is finished!")

    def read_epsg(self):
        ident = EpsgIdent()
        prj_file = self.path.replace('.shp', '.prj')
        ident.read_prj_from_file(prj_file)
        epsg = ident.get_epsg()
        return ('EPSG:' + str(epsg))

    def calculate_new_coordinates(self, center_pnt, width = 2500, height = 2500):
        # width and height variables must be in meters

        geod = pyproj.Geod(ellps='GRS80')

        rect_diag = sqrt( width**2 + height**2 )

        center_lon = -78.6389
        center_lat = 35.7806

        azimuth1 = atan(width/height)
        azimuth2 = atan(-width/height)
        azimuth3 = atan(width/height)+pi # first point + 180 degrees
        azimuth4 = atan(-width/height)+pi # second point + 180 degrees

        pt1_lon, pt1_lat, _ = geod.fwd(center_lon, center_lat, azimuth1*180/pi, rect_diag)
        pt2_lon, pt2_lat, _ = geod.fwd(center_lon, center_lat, azimuth2*180/pi, rect_diag)
        pt3_lon, pt3_lat, _ = geod.fwd(center_lon, center_lat, azimuth3*180/pi, rect_diag)
        pt4_lon, pt4_lat, _ = geod.fwd(center_lon, center_lat, azimuth4*180/pi, rect_diag)

        new_coordinates = ''
        
        return new_coordinates
