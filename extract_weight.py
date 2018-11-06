import h5py
import numpy as np

file = h5py.File('cifar10vgg.h5')
weights = []
for i in range(1,14):
    name = 'conv2d_{}'.format(i)
    w = file[name][name]['kernel:0'][()]
    w = w.transpose((3,2,1,0))
    weights.append(w.copy())
np.save('vgg16.npy', weights)