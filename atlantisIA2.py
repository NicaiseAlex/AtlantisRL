import imageio
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.utils
from sklearn.metrics import confusion_matrix, f1_score
import gym
import os
import cv2
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Reshape
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.convolutional import UpSampling2D, Convolution2D
from tensorflow.keras.layers import Dense, Activation, Permute, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dropout, Flatten

#Initialize
heightSizePictures = 40
widthSizePictures = 80
flattenImageSize = heightSizePictures * widthSizePictures
gamma = 0.99
update_frequency = 1
learning_rate = 0.001
resume = False
render = True
VERBOSE = False

env = gym.make('Atlantis-v4')
observation = env.reset()
number_of_inputs = env.action_space.n
prev_x = None
episode_number = 0
xs, dlogps, drs, probs = [],[],[],[]
running_reward = None
reward_sum = 0
train_X = []
train_y = []

def atlantis_model_checkpoint(I):
  I = I[0:80]
  I = I[::2,::2,0]
  #I[I == 144] = 0
  #I[I == 109] = 0
  #I[I != 0] = 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def modelDense(model_type=1):
    model = Sequential()

    if model_type == 0:
        model.add(Reshape((1,heightSizePictures,widthSizePictures), input_shape=(flattenImageSize,)))
        model.add(Flatten())
        model.add(Dense(200, input_dim=flattenImageSize, activation='sigmoid'))
        model.add(Dense(100, input_dim=200, activation='sigmoid'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = RMSprop(lr=learning_rate)
    else:
        model.add(Reshape((1,heightSizePictures,widthSizePictures,1), input_shape=(flattenImageSize,)))
        model.add(Convolution2D(32, 9, strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_uniform'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    if resume == True:
        model.load_weights('atlantis_model_checkpoint.h5')
    return model

modelToPLay = modelDense()

while True:
    if render:
        env.render()
    #Preprocess, consider the frame difference as features
    cur_x = atlantis_model_checkpoint(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(flattenImageSize)
    prev_x = cur_x
    #Predict probabilities from the Keras model
    aprob = (modelToPLay.predict(x.reshape([1, x.shape[0]]), batch_size=1).flatten())
    xs.append(x)
    probs.append((modelToPLay.predict(x.reshape([1, x.shape[0]]), batch_size=1).flatten()))
    aprob = aprob/np.sum(aprob)
    action = np.random.choice(number_of_inputs, 1, p=aprob)[0]
    y = np.zeros([number_of_inputs])
    y[action] = 1
    #print action
    dlogps.append(np.array(y).astype('float32') - aprob)
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    drs.append(reward)
    if done:
        episode_number += 1
        epx = np.vstack(xs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr
        #Slowly prepare the training batch
        train_X.append(xs)
        train_y.append(epdlogp)
        xs,dlogps,drs = [],[],[]
        #Periodically update the model
        if episode_number % update_frequency == 0:
            y_train = probs + learning_rate * np.squeeze(np.vstack(train_y)) #Hacky WIP
            #y_train[y_train<0] = 0
            #y_train[y_train>1] = 1
            #y_train = y_train / np.sum(np.abs(y_train), axis=1, keepdims=True)
            print('Training Snapshot:')
            print(y_train)
            modelToPLay.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)
            #Clear the batch
            train_X = []
            train_y = []
            probs = []
            #Save a checkpoint of the model
            os.remove('atlantis_model_checkpoint.h5') if os.path.exists('atlantis_model_checkpoint.h5') else None
            modelToPLay.save_weights('atlantis_model_checkpoint.h5')
        #Reset the current environment nad print the current results
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('Environment reset imminent. Total Episode Reward: %f. Running Mean: %f' % (reward_sum, running_reward))
        reward_sum = 0
        observation = env.reset()
        prev_x = None
    if reward != 0:
        print(('Episode %d Result: ' % episode_number) + ('Defeat!' if reward == -1 else 'VICTORY!'))