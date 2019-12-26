import shapefile    
import matplotlib.pyplot as plt
import cv2
from pyproj import Proj, transform
import os
from haversine import haversine, Unit
import numpy as np
from scipy import stats
import statistics as st


histogram_array = []

if os.path.isfile('./maximum_shapes_width.txt'):

    with open('maximum_shapes_width.txt', 'r+') as f:
        histogram_array = [float(x) for x in f.read().split()]

else:
    
    sf = shapefile.Reader(r"E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")

    print("Builting the histogram_array ")

    for shape_index, shape in enumerate(sf.shapes()): #loop shapes files
        
            
        xmin = shape.bbox[0]
        xmax = shape.bbox[2]
        ymin = shape.bbox[1]
        ymax = shape.bbox[3]

        # inProj = Proj(init='epsg:20791')
        # inProj = Proj(init='epsg:20790')
        inProj = Proj(init='epsg:3763')
        outProj = Proj(init='epsg:4326')
        x_minpoint = transform(inProj,outProj,xmin,ymin)
        x_maxpoint = transform(inProj,outProj,xmax, ymin)
        y_minpoint = transform(inProj,outProj,xmin,ymin)
        y_maxpoint = transform(inProj,outProj,xmin,ymax)

        x_dist = haversine(x_minpoint, x_maxpoint)
        y_dist = haversine(y_minpoint, y_maxpoint)

        if x_dist != "" and y_dist != "":
            if x_dist > y_dist:
                histogram_array.append(x_dist)
            else:
                histogram_array.append(y_dist)
        else:
            continue
        del x_dist, y_dist, xmin, xmax, ymin, ymax, x_minpoint, y_minpoint, x_maxpoint, y_maxpoint

    
# save data on text file
    with open('maximum_shapes_width.txt', 'w') as f:
        for item in histogram_array:
            f.write("%.16f\n" %item)


 
# An "interface" to matplotlib.axes.Axes.hist() method
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12, 8))


n, bins, patches = ax1.hist(x=histogram_array, bins=1000, color='#0504aa',
                            alpha=0.7, rwidth=0.85)

# tidy up the figure
average = sum(histogram_array) / len(histogram_array)
ax1.grid(True)
plt.text(3, 2, r'$\mu=%.3f km$' %(average))
plt.text(3, 1.8, r'$med=%.3f km$' %(st.median(histogram_array)))
ax1.legend(loc='right')
ax1.set_title('Number of Forest Fire and its Width 2017')
ax1.set_xlabel('Width (Km)',labelpad=0)
ax1.set_ylabel('Frequency')



# Cumulative histogram
n, bins, patches = ax2.hist(histogram_array, bins=100, density=True, histtype='step',
                           cumulative=True, label='Cumulative')

limit = 0.99

for i, value in enumerate(n):
    if value > limit:
        ind = i
        break
    else:
        continue

value_y = [limit] * (len(n) +1)
ax2.plot(bins, value_y, 'k--', linewidth=1.5, label='Y')
value_x = bins[ind]
plt.axvline(x=value_x, linewidth=1, color='black', linestyle='dashed')
plt.text(10, 0.9, r'$X=%.3f , Y=%.3f$' %(value_x, n[ind]))
# tidy up the figure
ax2.grid(True)
ax2.legend(loc='right')
ax2.set_title('Cumulative step histograms')
ax2.set_xlabel('Width (Km)')
ax2.set_ylabel('Cumulative Frequency')

plt.show()