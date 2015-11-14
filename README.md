# Image Recognizer with Chainer

## Requirement

* [Chainer](http://chainer.org/)
* [OpenCV](http://opencv.org/)
* Caffe Model  
  Download *.caffemodel from:
    * [bvlc_googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
* Some image files

## Example

```py
from model import GoogleNetModel
m = GoogleNetModel()
m.load("bvlc_googlenet.caffemodel")
m.load_label("labels.txt")
m.print_prediction("image.png")
```
