import gym
import cv2
import numpy as np
from gym.utils.play import play
import imageio

env = gym.make('Atlantis-v0')
env.reset()


def mycallback(obs_t, obs_tp1, action, rew, done, info):
    print("action :", action)
    imageio.imwrite('outfile.png', obs_t[0:160:4, ::2, 1])
    imageio.imwrite('outfile1.png', obs_tp1[0:160:4, ::2, 1])
    image1 = cv2.imread("outfile.png")
    image2 = cv2.imread("outfile1.png")
    image3 = image1[:,:, 1] - image2[:,:, 1]
    image3 = image3.reshape(1, (40*80))
    with open('X.txt', 'a') as outfileX:
        np.savetxt(outfileX, delimiter=',', X=image3, fmt="%s")
    with open('Y.txt', 'a') as outfileY:
        np.savetxt(outfileY, [action], delimiter=',', fmt="%s")


play(env, zoom=3, fps=12, callback=mycallback)
env.close()
