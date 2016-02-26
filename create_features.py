#!/usr/bin/env python
import os
import sys
# TODO: change this to your own caffe location
sys.path.append('/home/gen/caffe/python')
import argparse

from sklearn.externals import joblib
import caffe
import numpy as np
import scipy


# model setup - this will have different names depending on the model you download. It could be deploy.prototxt or solver.prototxt
MODEL_FILE = '/home/gen/caffe/models/hybridCNN/hybridCNN_deploy_FC7.prototxt'

# pretrained weights - this file will be called somthing.caffemodel
PRETRAINED = '/home/gen/caffe/models/hybridCNN/hybridCNN_iter_700000.caffemodel'

# Note: this feature normalization was recommended in Razavian et al. "Cnn features off-the-shelf: an astounding baseline for recognition."
def norm_convnet_feat(feature):
    D = 200
    try:
        feature = feature[0][:D]
    except IndexError, e:
        feature = feature[:D]
    
    # power norm
    alpha = 2.5

    feature = [np.power(f, alpha) if f > 0.0 else -1.0*np.power(np.abs(f), alpha) for f in feature]
    norm = np.linalg.norm(feature)
    feature = np.array([f/norm for f in feature])

    return feature


def img_feat(img_names, net, transformer, save_path, type_name, layer_name, normalize=True):

    for cnt, img_name in enumerate(img_names):
        print cnt

        if not os.path.exists(img_name):
            print 'skipping' 
            continue

        # this saves the features to different subdirs so not to many files end up in one dir. this can be turned off
        subdir = os.path.basename(img_name)[:2]
        if not os.path.exists(os.path.join(save_path, subdir)):
            os.makedirs(os.path.join(save_path, subdir))
            os.makedirs(os.path.join(save_path, subdir, 'norm'))            
        sname = os.path.join(save_path, subdir, '{0}_{1}.jbl'.format(os.path.basename(img_name).split('.')[0], layer_name))

        # check if this feature was already calculated
        if os.path.exists(sname):
            continue
        if not os.path.exists(os.path.join(save_path, subdir)):
            os.makedirs(os.path.join(save_path, subdir))
            
        img = scipy.misc.imread(img_name)
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1).repeat(3,2)        

        net.blobs['data'].data[...] = transformer.preprocess('data',img)
        out = net.forward(blobs=[layer_name])


        if normalize:
            feat = out[layer_name]
        else:
            feat = norm_convnet_feat(out[layer_name])

        # saving feature in joblib file format
        joblib.dump(feat, sname)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make caffe features for given image or patch")
    parser.add_argument("-i", "--image_name", help="", type=str, nargs='*')
    parser.add_argument("--feat_type", help="name of feature", type=str)    
    parser.add_argument("--model_file", help="caffe model params file", type=str)
    parser.add_argument("--pretrained", help="caffe pretrained model file", type=str)
    parser.add_argument("--layer_name", help="name of layer to extract feature from", type=str)
    parser.add_argument("--save_dir", help="location to save features", type=str)

    args = parser.parse_args()

    if args.model_file:
        MODEL_FILE = args.model_file
    if args.pretrained:
        PRETRAINED = args.pretrained
        
    net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
    # Note: this sets up a transformer object to change input images into the correct format for the pretrained model used
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))    
    transformer.set_mean('data', np.load('/home/gen/caffe/models/hybridCNN/hybridCNN_mean.npy').mean(1).mean(1))
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))
    net.blobs['data'].reshape(1,3,227,227)

    ###### TODO: comment out this line to use cpu only mode
    caffe.set_mode_gpu()
        
    img_feat(args.image_name, net, transformer, args.save_dir, args.feat_type, args.layer_name) 

