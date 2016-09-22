import sys
from level0_model import *

m = AlexNet()
if not m.load('bvlc_alexnet.pkl'):
    print 'Download bvlc_alexnet.caffemodel first.'
    print 'open https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet'
    exit -1

if not m.load_label('labels.txt'):
    print 'Make labels.txt first as:'
    print 'curl -O http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz'
    print "sed -e 's/^[^ ]* //g' synset_words.txt > labels.txt"
    exit -1

imagefile = 'image_vase.png' if len(sys.argv) < 2 else sys.argv[1]
m.print_prediction(imagefile)
