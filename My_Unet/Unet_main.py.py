# semantic segmentation with Unet
# heavily inspired on https://github.com/zhixuhao/unet

import os
import matplotlib.pyplot as plt
import skimage.io as skimage_io
import skimage
import skimage.transform as skimage_transform
from sklearn.metrics import classification_report, confusion_matrix
import random as r
import numpy as np
import tensorflow as tf
import random_folders_selection as rfs
from get_image_map_img import GetImageMap
import datetime
# print("Tensorflow version:",tf.__version__)
# print("GPU available:", tf.config.list_physical_devices('GPU'))
ts = datetime.datetime.now()

ts = ts.strftime("%d_%m_%Y_%H_%M_%S")


datasetPath = "C:\\Users\\GuilhermeRodrigues\\PycharmProjects\\tese_guilherme\\venv\\dataset_190_600"
modelsPath = "C:\\Users\\GuilhermeRodrigues\\PycharmProjects\\tese_guilherme\\venv\\models"
resultsPath = "C:\\Users\\GuilhermeRodrigues\\PycharmProjects\\tese_guilherme\\venv\\results\\test\\predict"

inputSize = (256, 256)
maskSize = (256, 256)
batchSize = 4
epochs = 100
learning_rate = 1e-4
numClasses = 14
showImages = False

modelFileName = "unet_membrane" + "E" + str(epochs) + "LR" + str(learning_rate) + "TS" + str(ts) + ".hdf5"

augmentation_args = dict(
    #width_shift_range=range(256),
    #height_shift_range=range(256),
    rotation_range=[0, 90, 180, 270],
    horizontal_flip=True,
    vertical_flip=True
)

def prepareDataset(datasetPath, trainFolder, valFolder, testFolder):
    trainSetX = []
    trainSetY = []
    valSetX = []
    valSetY = []
    testSetX = []

    trainImagesPath = os.path.join(datasetPath, trainFolder, "image")
    trainMasksPath = os.path.join(datasetPath, trainFolder, "label")
    print(os.getcwd())
    trainSetFolder = os.scandir(trainImagesPath)

    for tile in trainSetFolder:
        imagePath = tile.path
        trainSetX.append(imagePath)
        if (showImages):
            image =  skimage_io.imread(imagePath)
            maskPath = os.path.join(trainMasksPath, os.path.basename(imagePath))
            mask = skimage_io.imread(maskPath)
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image, cmap='gray')
            plt.xlabel("Image - {}".format(os.path.basename(imagePath)))
            plt.subplot(1, 2, 2)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(mask, cmap='gray')
            plt.xlabel("Mask")
            plt.show()

    r.shuffle(trainSetX)
    for trainExample in trainSetX:
        maskPath = os.path.join(trainMasksPath, os.path.basename(trainExample))
        trainSetY.append(maskPath)

    valImagesPath = os.path.join(datasetPath, valFolder, "image")
    # valImagesPath = os.path.join(datasetPath, trainFolder, "image")
    valSetXFolder = os.scandir(valImagesPath)
    for tile in valSetXFolder:
        imagePath = tile.path
        valSetX.append(imagePath)
    valMasksPath = os.path.join(datasetPath, valFolder, "label")
    # valMasksPath = os.path.join(datasetPath, trainFolder, "label")
    valSetYFolder = os.scandir(valMasksPath)
    for tile in valSetYFolder:
        maskPath = tile.path
        valSetY.append(maskPath)

    testImagesPath = os.path.join(datasetPath, testFolder, "image")
    # testImagesPath = os.path.join(datasetPath, testFolder)
    testSetFolder = os.scandir(testImagesPath)
    for tile in testSetFolder:
        imagePath = tile.path
        testSetX.append(imagePath)

    return trainSetX, trainSetY, valSetX, valSetY, testSetX


def normalizeMask(mask, num_class=2):
    mask = mask/255
    new_mask = np.zeros(mask.shape + (num_class,))
    for i in range(num_class):
        new_mask[mask == i, i] = 1.
    return new_mask


class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.batch_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracies.append(logs.get('accuracy'))

def normalizeChannel(channel, img_avg_std):

    return (channel - img_avg_std[0]) / img_avg_std[1]

def getImageChannels(tile, filename, img_avg_std):

    # Load the image.
    tile = os.path.join(tile, filename)
    img = skimage_io.imread(tile)
    #convert image to grayscale
    channel0 = skimage.util.img_as_ubyte(skimage.color.rgb2gray(img))
    #normalize image [-1, 1]
    channel0 = normalizeChannel(channel0, img_avg_std)
    return channel0

def getMaskChannels(tile, filename):

    # Load the image.
    tile = os.path.join(tile, filename)
    img = skimage.io.imread(tile)
    #normalize mask [0, 1]
    channel0 = normalizeMask(img)
    return channel0

def augmentImage(image, inputSize, mask, maskSize, aug_dict):

    if 'width_shift_range' in aug_dict:
        cropx = r.sample(aug_dict['width_shift_range'], 1)[0]
    else:
        cropx = (int)((image[0].shape[1] - inputSize[1]) / 2)
    if 'height_shift_range' in aug_dict:
        cropy = r.sample(aug_dict['height_shift_range'], 1)[0]
    else:
        cropy = (int)((image[0].shape[0] - inputSize[0]) / 2)
    if 'rotation_range' in aug_dict:
        rotation = r.sample(aug_dict['rotation_range'], 1)[0]
    else:
        rotation = 0
    if 'horizontal_flip' in aug_dict and aug_dict['horizontal_flip']:
        do_horizontal_flip = r.sample([False,True], 1)[0]
    else:
        do_horizontal_flip = False
    if 'vertical_flip' in aug_dict and aug_dict['vertical_flip']:
        do_vertical_flip = r.sample([False, True], 1)[0]
    else:
        do_vertical_flip = False

    maskOffsety = int((inputSize[0]-maskSize[0])/2)
    maskOffsetx = int((inputSize[1]-maskSize[1])/2)
    # mask = mask[maskOffsety+cropy:maskOffsety+cropy+maskSize[0], maskOffsetx+cropx:maskOffsetx+cropx+maskSize[1]]
    if rotation:
        mask = skimage_transform.rotate(mask, rotation)
    if do_horizontal_flip:
        mask = mask[:, ::-1]
    if do_vertical_flip:
        mask = mask[::-1, :]

    for i in range(image.shape[2]):
        channel = image[:,:,i]
        # channel = channel[cropy:cropy+inputSize[0], cropx:cropx+inputSize[1]]
        if rotation:
            channel = skimage_transform.rotate(channel, rotation)
        if do_horizontal_flip:
            channel = channel[:, ::-1]
        if do_vertical_flip:
            channel = channel[::-1, :]
        image[:,:,i] = channel

    return image, mask


def trainGenerator(batch_size, trainSet, avg_std, img_avg_std, aug_dict, inputSize=(256, 256), maskSize=(256, 256), numClasses=14):
    inputChannels = 13 + 1 + 8 #classes (cos  + highers + weather)

    if batch_size > 0:
        while 1:
            iTile = 0
            nBatches = int(np.ceil(len(trainSet)/batch_size))
            for batchID in range(nBatches):
                images = np.zeros(((batch_size,) + inputSize + (inputChannels,))) # 22 channels
                image_cube_all = np.zeros((inputSize + (inputChannels,))) # 22 channels
                masks = np.zeros(((batch_size,) + maskSize + (len(maskSize),)))
                iTileInBatch = 0
                while iTileInBatch<batch_size:
                    if iTile < len(trainSet):
                        # print(iTile, "/", len(trainSetX), ";", iTileInBatch, "/", batch_size, ";", trainSetX[iTile], trainSetY[iTile])
                        mapping = GetImageMap()
                        image_classed = mapping.BuildBinaryImage(os.path.join(trainSet[iTile], 'cos.png'), mapping.gc_cos)
                        image_cube_cos = rfs.BuildCOSCube(image_classed)
                        image_higher = getImageChannels(trainSet[iTile], 'higher.png', img_avg_std)
                        image_cube_all = rfs.addImagetoCube(image_cube_cos, image_higher)
                        weather_cube = rfs.build_weather_cube(trainSet[iTile], 'data.json', avg_std)
                        image_cube_all = rfs.Add2Cubes(inputSize, image_cube_all, weather_cube)
                        mask = getMaskChannels(trainSet[iTile], 'shape.png')

                        image, mask = augmentImage(image_cube_all, inputSize, mask, maskSize, aug_dict)
                        for i in range(image.shape[2]):
                            images[iTileInBatch, :, :, i] = image[:,:,i]
                        masks[iTileInBatch, :, :, :] = mask

                        iTile = iTile + 1
                        iTileInBatch = iTileInBatch + 1
                    else:
                        images = images[0:iTileInBatch,:,:,:]
                        masks = masks[0:iTileInBatch,:,:,:]
                        break
                yield (images, masks)


def unetCustom(pretrained_weights=None, inputSize=(256, 256, 22), numClass=2, do_batch_normalization=False, use_transpose_convolution=False):

    inputs = tf.keras.layers.Input(inputSize)

    Nchannels_COS = 14
    imgSize=inputSize[0:2]
    channels=inputSize[2]
    Nchannels_WEATHER=channels-Nchannels_COS
    full = tf.keras.layers.Reshape((inputSize + (1,)), input_shape=inputSize)(inputs)
    COS_channels= tf.keras.layers.Cropping3D(cropping=((0, 0), (0, 0), (0, Nchannels_WEATHER)))(full)
    WEATHER_channels = tf.keras.layers.Cropping3D(cropping=((0, 0), (0, 0), (Nchannels_COS, 0)))(full)
    COS_H = tf.keras.layers.Reshape((imgSize+ (Nchannels_COS,)), input_shape=(inputSize + (Nchannels_COS,)))(COS_channels)
    WEATHER_channels = tf.keras.layers.Reshape((imgSize+ (Nchannels_WEATHER,)), input_shape=(inputSize + (Nchannels_WEATHER,)))(WEATHER_channels)

    conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(COS_H)
    if do_batch_normalization:
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)
    conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    if do_batch_normalization:
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    if do_batch_normalization:
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    conv2 = tf.keras.layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    if do_batch_normalization:
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    if do_batch_normalization:
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)
    conv3 = tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    if do_batch_normalization:
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    if do_batch_normalization:
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('relu')(conv4)
    conv4 = tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    if do_batch_normalization:
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('relu')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    if do_batch_normalization:
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Activation('relu')(conv5)
    conv5 = tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    if do_batch_normalization:
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Activation('relu')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    if use_transpose_convolution:
        up6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2))(drop5)
    else:
        up6 = tf.keras.layers.Conv2D(512, 2, padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    if do_batch_normalization:
        up6 = tf.keras.layers.BatchNormalization()(up6)
    up6 = tf.keras.layers.Activation('relu')(up6)
    merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    if do_batch_normalization:
        conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Activation('relu')(conv6)
    conv6 = tf.keras.layers.Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    if do_batch_normalization:
        conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Activation('relu')(conv6)

    if use_transpose_convolution:
        up7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2))(conv6)
    else:
        up7 = tf.keras.layers.Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    if do_batch_normalization:
        up7 = tf.keras.layers.BatchNormalization()(up7)
    up7 = tf.keras.layers.Activation('relu')(up7)
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    if do_batch_normalization:
        conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Activation('relu')(conv7)
    conv7 = tf.keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    if do_batch_normalization:
        conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Activation('relu')(conv7)

    if use_transpose_convolution:
        up8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2))(conv7)
    else:
        up8 = tf.keras.layers.Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    if do_batch_normalization:
        up8 = tf.keras.layers.BatchNormalization()(up8)
    up8 = tf.keras.layers.Activation('relu')(up8)
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    if do_batch_normalization:
        conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Activation('relu')(conv8)
    conv8 = tf.keras.layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    if do_batch_normalization:
        conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Activation('relu')(conv8)

    if use_transpose_convolution:
        up9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv8)
    else:
        up9 = tf.keras.layers.Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
    if do_batch_normalization:
        up9 = tf.keras.layers.BatchNormalization()(up9)
    up9 = tf.keras.layers.Activation('relu')(up9)
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = tf.keras.layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    if do_batch_normalization:
        conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Activation('relu')(conv9)
    conv9 = tf.keras.layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    if do_batch_normalization:
        conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Activation('relu')(conv9)

    withWeather = tf.keras.layers.concatenate([conv9, WEATHER_channels], axis=3)
    conv10 = tf.keras.layers.Conv2D(numClass, 1, activation='softmax', kernel_initializer='he_normal')(withWeather)

    # conv10 = tf.keras.layers.Conv2D(numClass, 1, activation='softmax', kernel_initializer ='he_normal')(conv9)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def do_center_crop(image, newSize):
    cropy = (int)((image[0].shape[0] - newSize[0]) / 2)
    cropx = (int)((image[0].shape[1] - newSize[1]) / 2)
    for i in range(len(image)):
        channel = image[i]
        channel = channel[cropy:image[0].shape[0] - cropy, cropx:image[0].shape[1] - cropx]
        image[i] = channel

    return image


def testGenerator(testSetX, inputSize=(256, 256), inputChannels=1):

    #images = np.zeros(( inputSize + (inputChannels,)))  # 22 channels
    image_cube_all = np.zeros((inputSize + (inputChannels,)))  # 22 channels
    #masks = np.zeros(( inputSize + (len(inputSize),)))
    iTile = 0
    mapping = GetImageMap()

    for iTile in range(len(testSetX)):

        image_classed = mapping.BuildBinaryImage(os.path.join(testSetX[iTile], 'cos.png'), mapping.gc_cos)
        image_cube_cos = rfs.BuildCOSCube(image_classed)
        image_higher = getImageChannels(testSetX[iTile], 'higher.png', test_higher_avg_std)
        image_cube_all = rfs.addImagetoCube(image_cube_cos, image_higher)
        weather_cube = rfs.build_weather_cube(testSetX[iTile], 'data.json', test_avg_std)
        image_cube_all = rfs.Add2Cubes(inputSize, image_cube_all, weather_cube)
        image_cube_all = np.array([image_cube_all])
        yield (image_cube_all)

        iTile = iTile + 1


def saveResults(testSetX, results, resultsPath):
    y_gt = np.zeros((len(testSetX), 256, 256))
    y_predict = np.zeros((len(testSetX), 256, 256))
    black_px = 0
    white_px = 0
    total_white_px = 0
    total_black_px = 0
    for i, item in enumerate(results):
        filename = os.path.join(testSetX[i], 'shape.png')
        mask_predict = np.argmax(item, axis=-1)
        mask_predict = mask_predict.astype(np.uint8)
        mask_predict = mask_predict * 255
        real_mask = skimage.io.imread(filename)
        y_predict[i, :, :] = mask_predict
        y_gt[i, :, :] = real_mask
        filename = os.path.join(testSetX[i], 'shape_predict' + str(i) + '.png')
        skimage_io.imsave(os.path.join(resultsPath, os.path.basename(filename)), mask_predict)
        filename = os.path.join(testSetX[i], 'shape_real' + str(i) + '.png')
        skimage_io.imsave(os.path.join(resultsPath, os.path.basename(filename)), real_mask)
        white_px = np.count_nonzero(real_mask)
        black_px = ((256 * 256) - white_px)
        total_white_px = total_white_px + white_px
        total_black_px = total_black_px + black_px

        #print(f'Percentage of shape: {black_px / (256 * 256) * 100}%')

    print(f'White pixels total: {total_white_px}')
    print(f'Black pixels total: {total_black_px}')
    print(f'Percentage of shapes: {total_black_px / (total_white_px + total_black_px) * 100}%')
    return y_gt, y_predict

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def main():

    trainPaths, testPaths, valPaths = rfs.main(os.path.join(datasetPath, 'data.json'))

    batch_history = BatchLossHistory()

    #get averages and standard deviations of each file dataset
    train_avg_std = rfs.GetWeatherAvgStd(trainPaths, 'data.json') 
    val_avg_std = rfs.GetWeatherAvgStd(valPaths, 'data.json')
    global test_avg_std
    test_avg_std = rfs.GetWeatherAvgStd(testPaths, 'data.json')
    train_higher_avg_std = rfs.GetAverageAndStdv(trainPaths, 'higher.png')
    val_higher_avg_std = rfs.GetAverageAndStdv(valPaths, 'higher.png')    
    global test_higher_avg_std
    test_higher_avg_std = rfs.GetAverageAndStdv(testPaths, 'higher.png')

    trainGene = trainGenerator(batchSize, trainPaths, train_avg_std, train_higher_avg_std, augmentation_args, inputSize=inputSize,
                               maskSize=maskSize, numClasses=numClasses)
    valGene = trainGenerator(batchSize, valPaths, val_avg_std, val_higher_avg_std, dict(), inputSize=inputSize,
                             maskSize=maskSize, numClasses=numClasses)

    modelFilePath = os.path.join(modelsPath, modelFileName)
    model = unetCustom(inputSize=(256, 256, 22), numClass=2, do_batch_normalization=False,
                       use_transpose_convolution=False)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(modelFilePath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    log_dir = "logs_160_190\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    Ntrain = len(trainPaths)
    stepsPerEpoch = np.ceil(Ntrain / batchSize)
    Nval = len(valPaths)
    validationSteps = np.ceil(Nval / batchSize)

    history = model.fit(trainGene,
                        steps_per_epoch=stepsPerEpoch,
                        epochs=epochs,
                        callbacks=[model_checkpoint, batch_history, early_stopping_callback, tensorboard_callback],
                        validation_data=valGene,
                        validation_steps=validationSteps)

    #testPaths = rfs.AddFileToPath(testPaths, 'shape.png')
    testGene = testGenerator(testPaths, inputSize=inputSize, inputChannels=22)
    NTest = len(testPaths)
    # testSteps = np.ceil(NTest / batchSize)
    testSteps = np.ceil(NTest)
    results = model.predict(testGene, verbose=1)
    y_gt, y_predict = saveResults(testPaths, results, resultsPath)

    y_gt = y_gt.reshape(-1)
    y_predict = y_predict.reshape(-1)

    confusion_mat = confusion_matrix(y_gt, y_predict)  # , labels=[0,1])
    classification_rep = classification_report(y_gt, y_predict)  # , labels=[0,1])
    print(confusion_mat)
    print(classification_rep)

    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()

    # Plot training & validation loss values
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')

    plt.subplot(2, 2, 3)
    # plt.plot(moving_average(batch_history.batch_accuracies, 5))
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Batch')
    plt.legend(['Train'], loc='lower right')

    # Plot training & validation loss values
    plt.subplot(2, 2, 4)
    # plt.plot(moving_average(batch_history.batch_losses, 5))
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.legend(['Train'], loc='upper right')

    plt.show()


if __name__ == '__main__':
    main()
