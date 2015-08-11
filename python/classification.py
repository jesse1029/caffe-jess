import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    #!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet
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
    plt.show();


def main():
    caffe.set_mode_cpu()
    net = caffe.Net(caffe_root + 'models/bvlc_googlenet/deploy.prototxt',
                    caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel',
                    caffe.TEST)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
    # set net to batch size of 50
    #net.blobs['data'].reshape(50,3,227,227)
    
   # net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
   # out = net.forward()
   # print("Predicted class is #{}.".format(out['prob'].argmax()))
    
    #plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
    
    
    # load labels
    imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    except:
        #!../data/ilsvrc12/get_ilsvrc_aux.sh
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    
    # sort top k predictions from softmax output
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    print labels[top_k]
    
    # CPU mode
    net.forward()  # call once for allocation
    #%timeit net.forward()
    
    # GPU mode
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net.forward()  # call once for allocation
    #%timeit net.forward()
    
    #feat = net.blobs['pool5'].data[0]
    #vis_square(feat, padval=1)


if __name__ == '__main__':
    main()

