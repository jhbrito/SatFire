import shapefile    
import matplotlib.pyplot as plt

sf = shapefile.Reader(r"C:\Users\Guilhe5\Desktop\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06")

print("Initializing Display")
fig = plt.figure()
ax = fig.add_subplot(111)

print("Display Initialized")
shape = sf.shapeRecords()

plt.xlim([-500000, 500000])
plt.ylim([-400000, 400000])

# plt.xlim([sf.bbox[0], sf.bbox[2]])
# plt.ylim([sf.bbox[1], sf.bbox[3]])

for shape in sf.shapes():
    #print("Finding Points")
    points = shape.points
    #print("Found Points")    

    #print("Creating Polygon")
    ap = plt.Polygon(points, fill=False, edgecolor="k")
    ax.add_patch(ap)
    #print("Polygon Created")

print("Displaying Polygons")
# plt.show()

# plt.xlim([xmin, xmax])
# plt.ylim([ymin, ymax])

ax.set_axis_off() #do not show axis on image
plt.savefig('shape_all.png')

