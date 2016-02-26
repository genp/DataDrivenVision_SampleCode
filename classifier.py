#!/usr/bin/env python
import os, time, sys
import argparse

from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from app import db
from app.models import Label, Feature, Patch, Image, Annotation, ClassifierScore
from mturk import manage_hits

"""

Object for classifier classifier. 
Contains methods for creating, updating, and applying
classifier.

"""

class Classifier:
    def __init__(self):
        self.mdl = svm.LinearSVC()
        self.mdl.set_params(dual=True)
        self.mdl.set_params(C=1.0)
        self.mdl.set_params(verbose=True)
        self.mdl.set_params(class_weight='auto')

    def make_nonlinear(self):
        self.mdl = None
        self.mdl = svm.SVC()

    def get_params(self):
        return self.mdl.get_params()
    
    def train(self, train_set, train_lbls):
        tic = time.time()
        self.mdl.fit(train_set, train_lbls)
        print self.get_params() 
        print 'Training score: '+str(self.mdl.score(train_set, train_lbls))
        toc = time.time()-tic
        print 'Time elapsed: '+str(toc)

    def test(self, test_set):
        conf = self.mdl.decision_function(test_set)
        return conf

    def save(self, save_name):
        joblib.dump(self, save_name, compress=6)


# method for collecting features for given label
def get_examples(label_id, cat_id, feat_type, train_percent, use_whole_img=False):
    pos_patches = {}
    neg_patches = {}
    if cat_id == -1: # case for whole images, use all categories, etc.
        pos_patches['train'] = manage_hits.find_positives([label_id], [], [], 'train2014')
        neg_patches['train'] = manage_hits.find_positives([], [label_id], [], 'train2014')
        pos_patches['val'] = manage_hits.find_positives([label_id], [], [], 'val2014')
        neg_patches['val'] = manage_hits.find_positives([], [label_id], [], 'val2014')
    else:

        cat = Label.query.get(cat_id)
        if cat.parent_id == None:
            cat_ids = [x.id for x in Label.query.filter(Label.parent_id == cat_id).all()]
            print cat_ids
            pos_patches['train'] = []
            neg_patches['train'] = []
            pos_patches['val'] = []
            neg_patches['val'] = []
            for c in cat_ids:
                pos_patches['train'] += manage_hits.find_positives([label_id], [], c, 'train2014')
                neg_patches['train'] += manage_hits.find_positives([], [label_id], c, 'train2014')
                pos_patches['val'] += manage_hits.find_positives([label_id], [], c, 'val2014')
                neg_patches['val'] += manage_hits.find_positives([], [label_id], c, 'val2014')                
        else:
            pos_patches['train'] = manage_hits.find_positives([label_id], [], cat_id, 'train2014')
            neg_patches['train'] = manage_hits.find_positives([], [label_id], cat_id, 'train2014')
            pos_patches['val'] = manage_hits.find_positives([label_id], [], cat_id, 'val2014')
            neg_patches['val'] = manage_hits.find_positives([], [label_id], cat_id, 'val2014')            
        
    inter = set(pos_patches['train']).intersection(neg_patches['train'])
    for item in inter:
        pos_patches['train'].remove(item)
        neg_patches['train'].remove(item)
    inter = set(pos_patches['val']).intersection(neg_patches['val'])
    for item in inter:
        pos_patches['val'].remove(item)
        neg_patches['val'].remove(item)
                
    print len(pos_patches['train'])
    print len(neg_patches['train'])
    if len(pos_patches['train']) == 0 or len(neg_patches['train']) == 0:
        print 'either no positive or no negative training examples'
        return {}, len(pos_patches['train']), len(neg_patches['train']), []

    for tp in ['train', 'val']:
        pos_feat = []
        neg_feat = []
        missing_p = []
        for idx, p in enumerate(pos_patches[tp]):
            try:
                if use_whole_img:
                    img_id = Patch.query.get(p).image.id
                    feat = joblib.load(Feature.query.\
                                              filter(Feature.image_id == img_id).\
                                              filter(Feature.type == feat_type).\
                                              first().location)
                    print '%d img_id %d' % (idx, img_id)
                else: 
                    print '%d patch_id %d' % (idx, p)               
                    feat = joblib.load(Feature.query.\
                                              filter(Feature.patch_id == p).\
                                              filter(Feature.type == feat_type).\
                                              first().location)                
                if pos_feat == []:
                    pos_feat = feat
                else:
                    pos_feat = np.vstack([pos_feat, feat])                                      
            except AttributeError, e:
                missing_p.append(p)
                
        for idx, p in enumerate(neg_patches[tp]):
            try:
                if use_whole_img:
                    img_id = Patch.query.get(p).image.id
                    feat = joblib.load(Feature.query.\
                                              filter(Feature.image_id == img_id).\
                                              filter(Feature.type == feat_type).\
                                              first().location)
                    print '%d img_id %d' % (idx, img_id)                                                          
                else:
                    print '%d patch_id %d' % (idx, p)                                
                    feat = joblib.load(Feature.query.\
                                              filter(Feature.patch_id == p).\
                                              filter(Feature.type == feat_type).\
                                              first().location)                
                if neg_feat == []:
                    neg_feat = feat
                else:
                    neg_feat = np.vstack([neg_feat, feat])                                                                            
            except AttributeError, e:
                missing_p.append(p)

        if len(pos_feat) == 0 or len(neg_feat) == 0:
            print 'either no positive or no negative training examples'
            return {}, pos_feat.shape[0], neg_feat.shape[0], []
            
        if tp == 'train':        
            num_pos_train = pos_feat.shape[0]
            num_neg_train = neg_feat.shape[0]
            train = np.vstack([pos_feat, neg_feat])
            train_lbls = np.vstack([np.ones((num_pos_train,1)), -1*np.ones((num_neg_train,1))])
        else:
            num_pos_test = pos_feat.shape[0]
            num_neg_test = neg_feat.shape[0]
            test = np.vstack([pos_feat, neg_feat])        
            test_lbls = np.vstack([np.ones((num_pos_test,1)), -1*np.ones((num_neg_test,1))])    
                    
    examples = {}
    examples['train'] = train
    examples['train_lbls'] = train_lbls
    examples['test'] = test
    examples['test_lbls'] = test_lbls
    return examples, pos_feat.shape[0], neg_feat.shape[0], missing_p

# train classifier for given label        
def make_classifier(label_id, cat_id, feat_type, train_percent, spath, use_whole_img=False):
    cat = Label.query.get(cat_id)
    if cat == None:
        cat = 'all'
    else:
        cat = cat.name
    lbl = Label.query.get(label_id).name
    sname = os.path.join(spath, feat_type, '{0}_{1}.jbl'.format(cat,lbl))
    if os.path.exists(sname):
        print "The model at {} already exists :)".format(sname)
        return []
    examples, num_pos, num_neg, mp = get_examples(label_id, cat_id, feat_type, train_percent, use_whole_img)

    if examples == {}:
        return 0.0, 0.0, examples, num_pos, num_neg, mp
    
    cls = Classifier()
    cls.train(examples['train'], examples['train_lbls'])
    res = cls.test(examples['test'])

    ap_score = average_precision_score(examples['test_lbls'], res)
    acc = accuracy_score(examples['test_lbls'], [1 if x >=-0.0 else -1 for x in res])

    print 'AP score: %.4f' % ap_score
    print 'Accuracy (thresh = 0.0): %.4f' % acc

    cls.save(sname)

    return ap_score, acc, examples, num_pos, num_neg, mp 

def test_db(img_id, is_patch, attr_ids, cat_id, feat_type):
    if is_patch:
        feat = joblib.load(Feature.query.filter(Feature.patch_id == img_id).\
                                         filter(Feature.type == feat_type).\
                                         first().location)
    else:
        feat = joblib.load(Feature.query.filter(Feature.image_id == img_id).\
                                         filter(Feature.type == feat_type).\
                                         first().location)
    return test_feat(feat, attr_ids, cat_id, feat_type)

def test_image(img, attr_ids, cat_id, feat_type, layer_name):
    '''
    img is np.ndarray
    attr_ids are attributes to predict
    feat_type is feature name, corresponding to feature name in db
    layer_name is cnn layer to get activations from
    '''
    # model setup
    MODEL_FILE = '/home/gen/caffe/models/hybridCNN/hybridCNN_deploy_FC7.prototxt'
    # pretrained weights
    PRETRAINED = '/home/gen/caffe/models/hybridCNN/hybridCNN_iter_700000.caffemodel'
    # network setup
    net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load('/home/gen/caffe/models/hybridCNN/hybridCNN_mean.npy').mean(1).mean(1))
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))
    net.blobs['data'].reshape(1,3,227,227)

    ###### TODO: depends on machine
    caffe.set_mode_gpu()
    ######

    net.blobs['data'].data[...] = transformer.preprocess('data',img)
    out = net.forward(blobs=[layer_name])
    feat = out[layer_name]
    
    return test_feat(feat, attr_ids, cat_id, feat_type)
                                         
def test_feat(feat, attr_ids, cat_id, feat_type):                                         
    res = []
    for a in attr_ids:
        c = ClassifierScore.query.\
                            filter(ClassifierScore.type == feat_type).\
                            filter(ClassifierScore.label_id == a).\
                            filter(ClassifierScore.cat_id == cat_id).\
                            first()
        try:                    
            print Label.query.get(c.label_id).name
            print c.id
            mdl = joblib.load(c.location)
            conf = mdl.test(feat)
            res.append(conf)
        except (AttributeError, IOError), e:
            res.append(-10)
        
    return sorted(zip(attr_ids, res), key = lambda x: x[1], reverse=True)


def print_result(img, attr_confs, cat_id, sname):

    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')  # clear x- and y-axes
    if cat_id == -1:
        plt.title('all object classifier')
    else:
        plt.title(Label.query.get(cat_id).name)
    for ind, a in enumerate(attr_confs[:15]):
        attr = Label.query.get(a[0]).name
        t = '%s %0.3f' % (attr, a[1])
        print t
        plt.text(min(img.shape[1]+10, 1000), (ind+1)*img.shape[1]*0.1, t, ha='left')
    
    fig.savefig(sname, dpi = 300,  bbox_inches='tight')    
    pass    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train an svm")
    parser.add_argument("--location", help="", type=str)        
    parser.add_argument("--feat_type", help="name of feature", type=str)    
    parser.add_argument("--train_percent", help="", type=float)
    parser.add_argument("--cat_id", help="", type=int)
    parser.add_argument("--label_id", help="", type=int)
    parser.add_argument("--whole_img", help="", type=bool, default=False)

    args = parser.parse_args()

    try:
        ap_score, acc, examples, num_pos, num_neg, mp = make_classifier(args.label_id, args.cat_id,
                                                                        args.feat_type, args.train_percent,
                                                                        args.location, args.whole_img)        
        c = ClassifierScore(type=args.feat_type,
                            location=args.location,
                            cat_id=args.cat_id,
                            label_id=args.label_id,
                            num_pos=num_pos,
                            num_neg=num_neg,
                            train_percent=args.train_percent,
                            test_acc=acc,
                            ap=ap_score)
        db.session.add(c)
        db.session.commit()
    except ValueError, e:
        print 'model already trained.'
