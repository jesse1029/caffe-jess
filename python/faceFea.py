import numpy as np
import os, sys, getopt
 
# Main path to your caffe installation
caffe_root = '../'
 
# Model prototxt file
model_prototxt = '/home/jess/caffe/python/facedb/lenet.prototxt'
 
# Model caffemodel file
model_trained = '/home/jess/caffe/python/facedb/model/faceModel.caffemodel'
 
# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
 
# Path to the mean image (used for input processing)
mean_path = '/home/jess/caffe/python/facedb/imagenet_mean.binaryproto'
 
# Name of the layer we want to extract
layer_name = 'ip1'
 
sys.path.insert(0, caffe_root + 'python')
import caffe
 
def main(argv):
    inputfile = ''
    outputfile = ''
 
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'caffe_feature_extractor.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
 
    for opt, arg in opts:
        if opt == '-h':
            print 'caffe_feature_extractor.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg
 
    print 'Reading images from "', inputfile
    print 'Writing vectors to "', outputfile
 
    # Setting this to CPU, but feel free to use GPU if you have CUDA installed
    caffe.set_mode_gpu()
    print '1==================\n'
    # Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(model_prototxt, model_trained,
                           raw_scale=255,
                           image_dims=(80, 60))
    print '2==================\n'
    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()
    print '3'
    print "================================================"
    #print net.blobs[layer_name].data[0].length
    # This prints information about the network layers (names and sizes)
    # You can uncomment this, to have a look inside the network and choose which layer to print
    #print [(k, v.data.shape) for k, v in net.blobs.items()]
    #exit()
 
    # Processing one image at a time, printint predictions and writing the vector to a file
    with open(inputfile, 'r') as reader:
        with open(outputfile, 'w') as writer:
            writer.truncate()
            for image_path in reader:
                image_path = image_path.strip()
                input_image = caffe.io.load_image(image_path)
                prediction = net.predict([input_image], oversample=False)
                
                print os.path.basename(image_path), ' : ' , labels[prediction[0].argmax()].strip() , ' (', prediction[0][prediction[0].argmax()] , ')'
                np.savetxt(writer, net.blobs[layer_name].data[0].reshape(1,-1), fmt='%.8g')
                
 
if __name__ == "__main__":
    main(sys.argv[1:])