import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc

def split(x):
    assert type(x) == int
    t = int(np.floor(np.sqrt(x)))
    for a in range(t, 0, -1):
        if x % a == 0:
            return a, x / a


def grid_transform(x, size):
    a, b = split(x.shape[0])
    h, w, c = size[0], size[1], size[2]
    x = np.reshape(x, [a, b, h, w, c])
    x = np.transpose(x, [0, 2, 1, 3, 4])
    x = np.reshape(x, [a * h, b * w, c])
    if x.shape[2] == 1:
        x = np.squeeze(x, axis=2)
    return x


def grid_show(fig, x, size):
    ax = fig.add_subplot(111)
    x = grid_transform(x, size)
    if len(x.shape) > 2:
        ax.imshow(x)
    else:
        ax.imshow(x, cmap='gray')

def concat_multiple_images(data):
    num_images, height, width, channel = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
    x = int(np.sqrt(num_images))
    y = np.zeros([x*height, x*width, channel])
    for i in range(x):
        for j in range(x):
            y[i * height:(i+1) * height, j * width:(j+1)*width, :] = data[i + j * x]
    return y

