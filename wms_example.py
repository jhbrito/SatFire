import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
fig = plt.figure(figsize=(8,8))
latmin=36.5
lonmin=139
latmax=42
lonmax=142.5
m = Basemap(
llcrnrlat=latmin,urcrnrlat=latmax,\
llcrnrlon=lonmin,urcrnrlon=lonmax,\
lat_ts=20,resolution='i',epsg=4326)
m.wmsimage('http://www.finds.jp/ws/kiban25000wms.cgi?',xpixels=500,
layers=['PrefSmplBdr','RailCL'],styles=['thick',''])
m.wmsimage('http://www.finds.jp/ws/pnwms.cgi?',xpixels=500,
layers=['PrefName'],styles=['large'],transparent=True)
m.drawcoastlines()
plt.show()