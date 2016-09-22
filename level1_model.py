import sys, os
from level0_model import *
import chainer

# Thanks to http://qiita.com/tabe2314/items/6c0c1b769e12ab1e2614
def copy_model(src, dst):
    assert isinstance(src, chainer.link.Chain)
    assert isinstance(dst, chainer.link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, chainer.link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, chainer.link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print 'Copy %s' % child.name

# main
try:
    from VGGNet import *
except:
    print 'Download VGGNet.py'
    print 'curl -O https://raw.githubusercontent.com/mitmul/chainer-imagenet-vgg/master/VGGNet.py'
    exit -1

myModel = VGGNet()
rawModel = VGG19()
try:
    print 'Try to load VGG_ILSVRC_19_layers.npz'
    chainer.serializers.load_npz('VGG_ILSVRC_19_layers.npz', myModel)
    #print 'Try to load VGG_ILSVRC_19_layers_mitmul.npz'
    #chainer.serializers.load_npz('VGG_ILSVRC_19_layers_mitmul.npz', myModel)
except:
    if not rawModel.load('VGG_ILSVRC_19_layers.pkl'):
        print 'Download VGG_ILSVRC_19_layers.caffemodel first.'
        print 'open https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md'
        exit -1
    print 'Converting ...'
    copy_model(rawModel.func, myModel)
    chainer.serializers.save_npz('VGG_ILSVRC_19_layers.npz', myModel)
#except:
#    pass

if not rawModel.load_label('labels.txt'):
    print 'Make labels.txt first as:'
    print 'curl -O http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz'
    print "sed -e 's/^[^ ]* //g' synset_words.txt > labels.txt"
    exit -1

imagefile = 'image_vase.png' if len(sys.argv) < 2 else sys.argv[1]

# test by prediction
print 'Predicting ...'
image = rawModel.load_image(imagefile)
y = myModel(image, None)
result = zip(y.data.reshape((y.data.size,)), rawModel.labels)
prediction = sorted(result, reverse=True)
for i, (score, label) in enumerate(prediction[:10]):
    print '{:>3d} {:>6.2f}% {}'.format(i + 1, score * 100, label)

#rawModel.print_prediction(imagefile)
