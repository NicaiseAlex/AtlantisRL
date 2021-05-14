import imageio
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.utils
from sklearn.metrics import confusion_matrix, f1_score
import gym
import cv2
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import Dense, Activation, Permute, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dropout, Flatten


#Initialize
heightSizePictures = 40
widthSizePictures = 80
flattenImageSize = heightSizePictures * widthSizePictures

env = gym.make('Atlantis-v0')

def model():
    model = Sequential()

    #model.add(Dense(200, input_dim=flattenImageSize, activation='sigmoid'))
    #model.add(Dense(100, input_dim=200, activation='sigmoid'))

    model.add(Reshape((1, heightSizePictures, widthSizePictures, 1), input_shape=(flattenImageSize,)))
    model.add(Convolution2D(32, 9, strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))

    model.add(Dense(nbClasses, activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print("xTrain : ", xTrain.shape)
    print("xTest : ", xTest.shape)
    print("yTrain : ", yTrain.shape)
    print("yTest : ", yTest.shape)

    ourCallback = EarlyStopping(monitor='val_accuracy', min_delta = 0.0001, patience = 20, verbose = 0, mode ='auto', baseline = None, restore_best_weights = False)
    model.fit(xTrain, yTrain, epochs=1000, batch_size=128, validation_split=0.2, callbacks=[ourCallback])
    #model.fit(xTrain, yTrain, epochs=1000, batch_size=128, validation_split=0.2)


    score = model.evaluate(xTest, yTest)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    pred_test = np.argmax(model.predict(xTest), axis=1)
    print(pred_test.shape, np.argmax(yTest, axis=1).shape)
    print("F1 score: ", f1_score(pred_test, np.argmax(yTest, axis=1), average=None))
    print("F1 score micro: ", f1_score(pred_test, np.argmax(yTest, axis=1), average='micro'))
    print("F1 score macro: ", f1_score(pred_test, np.argmax(yTest, axis=1), average='macro'))
    print('confusion matrix\n', confusion_matrix(np.argmax(yTest, axis=1), pred_test))

    return model

def play():

    modelToPLay = model()

    move = 0

    obs = env.reset()
    obs1 = 0
    count = 0
    while True:
        env.render()
        obs, rew, d, inf = env.step(move)  # take a random action
        imageio.imwrite('outfile.png', obs[0:160:4, ::2, 1])
        if (count == 0):
            obs1 = obs
            count = count + 1
        imageio.imwrite('outfile1.png', obs1[0:160:4, ::2, 1])
        image1 = cv2.imread("outfile.png")
        image2 = cv2.imread("outfile1.png")
        image3 = image1[:, :, 1] - image2[:, :, 1]
        image3 = image3.reshape(1, (40 * 80))
        move = np.argmax(modelToPLay.predict(image3), axis=1)
        obs1 = obs
        if rew != 0:
            print("reward: ", rew)

    env.close()

with open('X.txt', 'r') as outfileX:
    allXdata = np.loadtxt(outfileX, delimiter=',')
with open('Y.txt', 'r') as outfileY:
    allYdata = np.loadtxt(outfileY, delimiter=',')

shuffle(allXdata, allYdata, random_state=0)
amountOfData = len(allYdata)
trainPercentage = 0.7
trainAmount = int(amountOfData * trainPercentage)
print("Nb data : ", amountOfData)
print("Nb data for train : ", trainAmount)
print("Nb data for test : ", amountOfData - trainAmount)
xTrain = allXdata[0:trainAmount, :]
y_Train = allYdata[0:trainAmount]
xTest = allXdata[trainAmount:amountOfData, :]
y_Test = allYdata[trainAmount:amountOfData]
y_Train = y_Train.reshape(trainAmount)
y_Test = y_Test.reshape((amountOfData - trainAmount))
print("Shape xTrain : ", xTrain.shape)
print("Shape xTest : ", xTest.shape)
print("Shape yTrain : ", y_Train.shape)
print("Shape yTest : ", y_Test.shape)
nbClasses = 4
xTrain = xTrain.astype('float32') / 255
xTest = xTest.astype('float32') / 255
y_Train = y_Train.astype('uint8')
y_Test = y_Test.astype('uint8')

yTrain = tensorflow.keras.utils.to_categorical(y_Train, None)
yTest = tensorflow.keras.utils.to_categorical(y_Test, None)

play()