import shapefile    
import matplotlib.pyplot as plt
import cv2
from pyproj import Proj, transform
import os
from haversine import haversine, Unit

# sf = shapefile.Reader(r"C:\Users\Guilhe5\Desktop\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")
sf = shapefile.Reader(r"E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")

print("Initializing Display")
fig = plt.figure()
ax = fig.add_subplot(111)

print("Display Initialized")
shapes = sf.shapes()
points = shapes[1197].points #select the shape we want see
ap = plt.Polygon(points, fill=True, edgecolor="k")
ax.add_patch(ap)

xmin = 9999999
xmax = -9999999
ymin = 9999999
ymax = -9999999

for index, (pointx, pointy) in enumerate(points):
    
    if index == 0:
        xmin = pointx
        xmax = xmin
    else:    
        if xmin > pointx:
           xmin = pointx 

        if xmax < pointx:
           xmax = pointx

        if ymin > pointy:
           ymin = pointy 

        if ymax < pointy:
           ymax = pointy

plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])


# inProj = Proj(init='epsg:20791')
inProj = Proj(init='epsg:3763')
outProj = Proj(init='epsg:4326')
x_minpoint = transform(inProj,outProj,xmin,ymin)
x_maxpoint = transform(inProj,outProj,xmax, ymin)
y_minpoint = transform(inProj,outProj,xmin,ymin)
y_maxpoint = transform(inProj,outProj,xmin,ymax)

x_dist = haversine(x_minpoint, x_maxpoint)
y_dist = haversine(y_minpoint, y_maxpoint)
print(x_minpoint)
print(x_maxpoint)
print(y_minpoint)
print(y_maxpoint)
print(y_dist)

print("Displaying Polygons")

# Create target Directory if don't exist
dirName = 'Image_1'
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")

ax.set_axis_off() #do not show axis on image
# plt.savefig('shape1.png')
plt.show()
# # convert image to binary
# img = cv2.imread('shape1.png',2)
# ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# cv2.imshow("Binary Image",bw_img)
# cv2.imwrite('shape1_bin.png', bw_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # resize image
# dim = (256, 256)
# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# cv2.imwrite('shape1_bin_resized.png', bw_img)
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

