import numpy as np
import matplotlib.pyplot as plt


import sys
caffe_root = '../' 
sys.path.insert(0, caffe_root + 'python')

import caffe

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    

    plt.imshow(data)
    plt.show()

def main():
    # Setting for plot function
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                       caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

    # do the prediction task
    scores = net.predict([caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')])
    
    # Retrieve the feature at conv1
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0,2,3,1))


if __name__ == '__main__':
    main()

