import shapefile   
import matplotlib.pyplot as plt
import cv2
from pyproj import Proj, transform
import os

# print("This is my file to demonstrate best practices.")

def save_images():

# read all shapefile
    sf = shapefile.Reader(r"C:\Users\Guilhe5\Desktop\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")
    for shape in sf.shapes():

#print("Finding Points")
        points = shape.points  

        xmin = 9999999
        xmax = -9999999
        ymin = 9999999
        ymax = -9999999

# build polygon shape
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

        xmin = xmin - 100
        xmax = xmax + 100
        ymin = ymin - 100
        ymax = ymax + 100

        inProj = Proj(init='epsg:20791')
        outProj = Proj(init='epsg:4326')
        x_minpoint = transform(inProj,outProj,xmin,ymin)
        x_maxpoint = transform(inProj,outProj,xmax, ymin)
        y_minpoint = transform(inProj,outProj,xmin,ymin)
        y_maxpoint = transform(inProj,outProj,xmin,ymax)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

# convert coordinates to lat/long
        ap = plt.Polygon(points, fill=True, edgecolor="k")
        inProj = Proj(init='epsg:20791')
        outProj = Proj(init='epsg:4326')

        lat1,long1 = transform(inProj,outProj,ymin,xmin)
        lat2,long2 = transform(inProj,outProj,ymax,xmax)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off() #do not show axis on image
        plt.savefig('shape1.png')

# convert image to binary and save it
        img = cv2.imread('shape1.png',2)
        ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        cv2.imshow("Binary Image",bw_img)
        path = r"C:\Users\Guilhe5\Desktop\Tese\Scripts\Imagens"
        cv2.imwrite(os.path.join(path , 'shape1_bin.png'), bw_img)
        cv2.waitKey(0)


def main():
    save_images()

if __name__ == "__main__":
    main()