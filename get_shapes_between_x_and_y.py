import shapefile    
from pyproj import Proj, transform
from haversine import haversine


class DeleteExtraShapes:

    def __init__(self, len_min, len_max, path):
        self.min = len_min
        self.max = len_max
        self.path = path
        
        self.inProj = Proj(init='epsg:3763')
        self.outProj = Proj(init='epsg:4326')

    def delete_shapes_out_of_range(self):
               
        sf = shapefile.Reader(r"E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")
        
        print("Builting the new shape file ")
        w = shapefile.Writer('E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\copy')
        w.fields = sf.fields[1:] # skip first deletion field
        count = 0
        total = 0
        for shape in sf.iterShapeRecords(): #loop shapefile
            total = total + 1

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
                if ((x_dist <= 5.120 and x_dist >= 0.040) or (y_dist <= 5.120 and y_dist >= 0.040) ):
                    w.record(*shape.record)
                    w.shape(shape.shape)
                    count = count + 1
                else:
                    continue
            else:
                continue
            del x_dist, y_dist, xmin, xmax, ymin, ymax, x_minpoint, y_minpoint, x_maxpoint, y_maxpoint

        w.close()

        print("Number of shapes saved: %i" %count)
        print("Total of shapes: %i" %total)
        print("->File Finished!<-")