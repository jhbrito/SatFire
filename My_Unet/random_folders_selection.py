import json
import random
import os
from PIL import Image
import numpy as np
import json
from datetime import date

def ReadJson(path = r'E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\new_shape\dataset\data.json'):

    if os.path.exists(path):
        data = json.load(open(path))
    else:
        raise Exception ('Json file does not exist!!!!')

    return data

def ReadMaxAndMinValue(json_data={}):
    area_max = 0.0
    area_min = 99999999.99

    for key, element in json_data.items():

        if float(element['area']) > area_max:
            area_max = float(element['area'])

        if float(element['area']) < area_min:
            area_min = float(element['area'])
        continue

    return area_max, area_min

def BuildClassesList(max_val, min_val, number_classes = 8):
    classes_list=[]
    val = min_val

    jump = float((max_val - min_val)/number_classes)

    while val <= max_val:
        classes_list.append(val)
        val = val + jump

    return classes_list

def BuildClassesGroups(json_data, classes_list):
    classes_dic={}
    folders_list=[]
    index_old = 0

    for index, classe in enumerate(classes_list):
        for key, element in json_data.items():

            if index != index_old and index != len(classes_list) - 1:
                index_old = index
                folders_list = []

            if index < (len(classes_list) - 1):
                if float(element['area']) >= classes_list[index] and float(element['area']) < classes_list[index+1]:
                    # Adding a new key value pair
                    folders_list.append(key)
                    classes_dic.update( {index+1:folders_list} )

            if index == len(classes_list) - 1:
                folders_list.append(list(reversed(list(json_data)))[0])
                classes_dic.update( {index:folders_list} )
                break

    return classes_dic

def GetRandomFolders(classes_dic, perc_train=60, perc_test=20, perc_val=20):
    folders_dic={}
    random.seed(4) #to generate always the some output when shuffle the numbers list
    test_folders=[]
    train_folders=[]
    val_folders=[]

    for index, (key, element) in enumerate(classes_dic.items()):

        numbers_list = element
        random.shuffle(numbers_list)

        nr_train = round(len(numbers_list) * (perc_train/100))
        nr_test = round(len(numbers_list) * (perc_test/100))
        nr_val = round(len(numbers_list) * (perc_val/100))


        val_folders = val_folders + numbers_list[:(nr_val)]
        del(numbers_list[:(nr_val)])

        test_folders = test_folders + numbers_list[:(nr_test)]
        del(numbers_list[:(nr_test)])

        train_folders = train_folders + numbers_list

    return train_folders, test_folders, val_folders

def GetFoldersPath(path, train_folders, test_folders, val_folders):
    trainPaths = []
    testPaths = []
    valPaths = []
    # path = "E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\new_shape\dataset"

    for folder in train_folders:
        path = os.path.join(path, folder)
        trainPaths.append(path)
        head_tail = os.path.split(path)
        path = head_tail[0]

    for folder in test_folders:
        path = os.path.join(path, folder)
        testPaths.append(path)
        head_tail = os.path.split(path)
        path = head_tail[0]

    for folder in val_folders:
        path = os.path.join(path, folder)
        valPaths.append(path)
        head_tail = os.path.split(path)
        path = head_tail[0]

    return trainPaths, testPaths, valPaths

def GetAverageAndStdv(ListDirectory, filename):

    total_images = int(len(ListDirectory))
    total_avg = 0
    total_stdv = 0
    for directory in ListDirectory:
        image_path = os.path.join(directory, filename)
        if os.path.exists(image_path):
            avg, stdv = CalculateImageAverageStdv(image_path)
            total_avg = total_avg + avg
            total_stdv = total_stdv + stdv
        elif image_path in 'data.json':
            continue
        else:
            raise Exception('Image File does not exist or directory path incorrect!')

    e_average = total_avg / total_images
    e_stdv = total_stdv / total_images

    return e_average, e_stdv

def CalculateImageAverageStdv(image_path):

    im = Image.open(image_path).convert('L')

    # Map PIL mode to numpy dtype (note this may need to be extended)
    dtype = {'F': np.float32, 'L': np.uint8}[im.mode]

    # Load the data into a flat numpy array and reshape
    np_img = np.array(im.getdata(), dtype=dtype)
    w, h = im.size
    np_img.shape = (h, w, np_img.size // (w * h))

    average = np.average(np_img)
    std = np.std(np_img)
    return average, std

def AddFileToPath(folder_path, filename):

    for index, directory in enumerate(folder_path):
        folder_path[index] = os.path.join(directory, filename)

    return folder_path

def BuildCOSCube(image, nClasses=13):

    n=1
    images = np.zeros((image.shape + (nClasses,)))

    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]

    bin_image = np.zeros(image.shape, dtype=np.uint8)

    while n <= nClasses:
        bin_image[image==n] = 1
        # for y in range(0, h):
        #     for x in range(0, w):
        #         if image[x,y] == n:
        #             bin_image[x,y]=1
        #         else:
        #             bin_image[x,y]=0

        images[ :, :, n-1] = bin_image
        n += 1
    return images

def addImagetoCube(cube, image):

    o_image = np.zeros((image.shape + (cube.shape[2]+1,)))
    i=0
    for i in range(cube.shape[2]):

        o_image[:,:,i] = cube[:,:,i]

    o_image[:,:,i+1] = image

    return o_image

def build_weather_cube(directories, filename, avg_std):

    o_image = np.zeros(((256,256) + (8,)), dtype=int)
    main_dir, folder = os.path.split(directories)
    json_data = {}
    json_file = os.path.join(main_dir, filename)
    i= 0
    with open(json_file) as j:
        json_data = json.load(j)

    for key, value in json_data[folder].items():
        if key == 'date':
            year = int(value[:4])
            month = int(value[5:7])
            day = int(value[8:11])
            mat_value = date(year,month,day) - date(year,1,1)
            mat_value = float((float(mat_value.days) - float(avg_std[0][0])) / float(avg_std[0][1])) #normalize
        elif key in ('hour'):
            mat_value = float((float(value) - float(avg_std[1][0])) / float(avg_std[1][1])) #normalize
        elif key in ('humidity'):
            mat_value = float((float(value) - float(avg_std[2][0])) / float(avg_std[2][1])) #normalize
        elif key in ('tempC'):
            mat_value = float((float(value) - float(avg_std[3][0])) / float(avg_std[3][1])) #normalize
        elif key in ('windspeedKmph'):
            mat_value = float((float(value) - float(avg_std[4][0])) / float(avg_std[4][1])) #normalize
        elif key in ('precipMM'):
            mat_value = float((float(value) - float(avg_std[5][0])) / float(avg_std[5][1])) #normalize
        elif key in ('cloudcover'):
            mat_value = float((float(value) - float(avg_std[6][0])) / float(avg_std[6][1])) #normalize
        elif key in ('WindGustKmph'):
            mat_value = float((float(value) - float(avg_std[7][0])) / float(avg_std[7][1])) #normalize
        else:
            continue

        matrix = np.ones((256,256),dtype = int) * int(round(float(mat_value),0))
        o_image[:,:,i] = matrix
        i+=1

    return o_image

def Add2Cubes(size, cube1, cube2):

    o_image = np.zeros((size + (cube1.shape[2] + cube2.shape[2],)))
    i=0

    for i in range(cube1.shape[2]):

        o_image[:,:,i] = cube1[:,:,i]
    n = i + 1
    i=0

    for i in range(cube2.shape[2]):

        o_image[:,:,n] = cube2[:,:,i]
        n+=1

    return o_image

def GetWeatherAvgStd(path_dir, datafile):

    total_days = np.array([])
    total_hour = np.array([])
    total_humidity = np.array([])
    total_tempC = np.array([])
    total_windspeedKmph = np.array([])
    total_precipMM = np.array([])
    total_cloudcover = np.array([])
    total_WindGustKmph = np.array([])

    path, folder = os.path.split(path_dir[0])
    json_file = os.path.join(path, datafile)
    json_data = {}
    with open(json_file) as j:
        json_data = json.load(j)

    for directory in path_dir:
        path, folder = os.path.split(directory)

        for key, value in json_data[folder].items():
            if key == 'date':
                year = int(value[:4])
                month = int(value[5:7])
                day = int(value[8:11])
                mat_value = date(year,month,day) - date(year,1,1)
                total_days = np.append(total_days, mat_value.days)
            elif key in ('hour'):
                total_hour = np.append(total_hour, float(value))
            elif key in ('humidity'):
                total_humidity = np.append(total_humidity, float(value))
            elif key in ('tempC'):
                total_tempC = np.append(total_tempC, float(value))
            elif key in ('windspeedKmph'):
                total_windspeedKmph = np.append(total_windspeedKmph, float(value))
            elif key in ('precipMM'):
                total_precipMM = np.append(total_precipMM, float(value))
            elif key in ('cloudcover'):
                total_cloudcover = np.append(total_cloudcover, float(value))
            elif key in ('WindGustKmph'):
                total_WindGustKmph = np.append(total_WindGustKmph, float(value))
            else:
                continue

    return(
            [(np.average(total_days),np.std(total_days)),
            (np.average(total_hour),np.std(total_hour)),
            (np.average(total_humidity),np.std(total_humidity)),
            (np.average(total_tempC),np.std(total_tempC)),
            (np.average(total_windspeedKmph),np.std(total_windspeedKmph)),
            (np.average(total_precipMM),np.std(total_precipMM)),
            (np.average(total_cloudcover),np.std(total_cloudcover)),
            (np.average(total_WindGustKmph),np.std(total_WindGustKmph))]
            )      

def main(json_path, perc_train=60, perc_test=20, perc_val=20):

    data = ReadJson(json_path)
    max_val, min_val = ReadMaxAndMinValue(data)
    classes_list = BuildClassesList(max_val, min_val)
    classe_groups = BuildClassesGroups(data, classes_list)
    train_folders, test_folders, val_folders = GetRandomFolders(classe_groups, perc_train, perc_test, perc_val)
    path, jsonname = os.path.split(json_path)
    trainPaths, testPaths, valPaths = GetFoldersPath(path, train_folders, test_folders, val_folders)

    return trainPaths, testPaths, valPaths

# if __name__ == "__main__":

    # data = ReadJson(r'E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\new_shape\dataset\data.json')
    # max_val, min_val = ReadMaxAndMinValue(data)
    # classes_list = BuildClassesList(max_val, min_val)
    # classe_groups = BuildClassesGroups(data, classes_list)
    # train_folders, test_folders, val_folders = GetRandomFolders(classe_groups)
    # trainPaths, testPaths, valPaths = GetFoldersPath(train_folders, test_folders, val_folders)
    # avg, stdv = GetAverageAndStdv(r'E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\new_shape\dataset', 'cos.tiff')




