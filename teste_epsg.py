import shapefile    
from pyproj import Transformer

sf = shapefile.Reader(r"E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")

transformer = Transformer.from_crs(20790, 4326, always_xy=True,)

points = [(241798, 164274)]

for pt in transformer.itransform(points): print('{:.3f} {:.3f}'.format(*pt))
