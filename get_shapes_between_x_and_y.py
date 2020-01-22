import shapefile    
from pyproj import Proj, transform
from haversine import haversine
from get_image_map import GetImageMap
from epsg_ident import EpsgIdent
import shutil

class DeleteExtraShapes:

    def __init__(self, len_min, len_max, path):
        self.min = len_min
        self.max = len_max
        self.path = path

        self.inProj = Proj(init='epsg:3763')
        self.outProj = Proj(init='epsg:4326')

        self.url_water = 'https://inspire.apambiente.pt/getogc/services/INSPIRE/AM_WaterBodyForWFD_WFDRiver/MapServer/WMSServer?SERVICE=WMS&REQUEST=GetCapabilities'
        self.url_vegetation = 'http://mapas.dgterritorio.pt/wms/hrl-PTContinente'

        self.epsg = self.read_epsg()

    def delete_shapes_out_of_range(self):
               
        # sf = shapefile.Reader(r"E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")
        sf = shapefile.Reader(self.path)
        
        print("Builting the new shape file ")
        new_doc = 'E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\copy'
        w = shapefile.Writer(new_doc)
        w.fields = sf.fields[1:] # skip first deletion field

        for shape in sf.iterShapeRecords(): #loop shapefile
            del x_dist, y_dist, xmin, xmax, ymin, ymax, x_minpoint, y_minpoint, x_maxpoint, y_maxpoint
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
            
        w.close()

        # create the PRJ file
        prj_file_old = self.path.replace('.shp', '.prj')
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

