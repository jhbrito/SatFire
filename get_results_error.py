
import os
import skimage.io as skimage_io
import numpy as np
import matplotlib.pyplot as plt

path = r'E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\resultados'
image_folder = 'predict'
gc_real = 'real'

row = 0
col = 0 
perc_array = []
perc_matr = []
accuracy_array = []
accuracy_matr = []

# A function that returns the length of the value:
def myFunc(e):
  return len(e) 


def calculate_average_area(val1, val2):
    if val1 <= val2:
        perc = (val1*100)/val2
    else:
        perc = (val2*100)/val1
    
    return perc


entries = os.listdir(path)

for entry in entries:
    folder_path = os.path.join(path, entry)
    folder_path = os.path.join(folder_path, image_folder)
    images = os.listdir(folder_path)
    images.sort(reverse=False, key=myFunc)
    images.reverse()
    
    for image in images: 
        if image_folder in image:    
            #predict   
            image_path = os.path.join(folder_path, image)
            img_predict = skimage_io.imread(image_path)
            matrix_predict = np.matrix(img_predict)
            black_pixels_predict = (img_predict.shape[0]*img_predict.shape[1]) - (matrix_predict.sum()/255)
            
            #real
            image_path = os.path.join(folder_path, image.replace(image_folder, gc_real))
            img_real = skimage_io.imread(image_path)
            matrix_real = np.matrix(img_real)
            black_pixels_real = (img_real.shape[0]*img_real.shape[1]) - (matrix_real.sum()/255)
            
            #error
            error = abs(black_pixels_predict - black_pixels_real)
            error_perc = (100*error)/(img_real.shape[0]*img_real.shape[1])
            perc_array.append(error_perc)

            #accuracy
            perc_accuracy = calculate_average_area(black_pixels_predict, black_pixels_real)
            accuracy_array.append(perc_accuracy)

        else:
            print(f'{np.mean(perc_array)}% +- {np.std(perc_array)}%')
            perc_matr.append(perc_array)
            perc_array=[]
            accuracy_matr.append(accuracy_array)
            accuracy_array=[]
            break

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5)
ax1.plot(accuracy_matr[0])
ax2.plot(accuracy_matr[1])
ax3.plot(accuracy_matr[2])
ax4.plot(accuracy_matr[3])
ax5.plot(accuracy_matr[4])
ax6.plot(accuracy_matr[5])
ax7.plot(accuracy_matr[6])
ax8.plot(accuracy_matr[7])
ax9.plot(accuracy_matr[8])
ax10.plot(accuracy_matr[9])

plt.show()