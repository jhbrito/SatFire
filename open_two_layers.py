from owslib.wms import WebMapService
from PIL import Image
import os

wms_1 = WebMapService('http://mapas.dgterritorio.pt/wms/hrl-PTContinente', version='1.3.0')
# type = wms_1.identification.type
# print(wms_1.identification.type)
# print(wms_1.identification.title)
# print(list(wms_1.contents))
# print(wms_1['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].title)
# print(wms_1['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].queryable)
# print(wms_1['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].opaque)
# print(wms_1['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].boundingBox)
# print(wms_1['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].boundingBoxWGS84)
# print(wms_1['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].crsOptions)
# print(wms_1['HRL_Zonas_Humidas_e_Corpos_Agua_2015'].styles)
# print(wms_1.getOperationByName('GetMap').methods)
# print(wms_1.getOperationByName('GetMap').formatOptions) 

response_1 = wms_1.getmap( layers=['HRL_Zonas_Humidas_e_Corpos_Agua_2015'],
                    styles=['default'],
                    srs='EPSG:3763',
                    bbox=(-241375.0, -366026.0, 282525.0, 342374.0),
                    size=(400, 800),
                    format='image/png',)                   
out_1 = open('first_layer.png', 'wb')
out_1.write(response_1.read())
out_1.close()

wms_2 = WebMapService('http://si.icnf.pt/wms/ardida_2017?service=wms&version=1.1.1&request=GetCapabilities', version='1.1.1')
response_2 = wms_2.getmap( layers=['ardida_2017'],
                    styles=['BDG:ardida_2017'],
                    srs='EPSG:3763',
                    bbox=(-111304.9921875, -291967.625, 160375.421875, 273254.84375),
                    size=(400, 800),
                    format='image/png',
                    transparent=False)

out_2 = open('second_layer.png', 'wb')
out_2.write(response_2.read())
out_2.close()

#  #Relative Path 
#  #Image on which we want to paste 
# img = Image.open("first_layer.jpeg")  
          
# #Relative Path 
# #Image which we want to paste 
# img2 = Image.open("second_layer.jpeg")  
# img.paste(img2, (200, 400)) 
          
# #Saved in the same relative location 
# img.save("pasted_picture.jpeg") 
filename = 'second_layer.png'
ironman = Image.open(filename, 'r')
width, height = ironman.size
print(width)
print(height)
filename1 = 'first_layer.png'
first_layer = Image.open(filename1, 'r')
text_img = Image.new('RGBA', (400,800), (0, 0, 0, 0))
text_img.paste(first_layer, (0,0))
text_img.paste(ironman, (0,0), mask=ironman)
text_img.save("ball.png", format="png")