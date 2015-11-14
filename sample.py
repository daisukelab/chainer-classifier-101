from model import GoogleNetModel

m = GoogleNetModel()
m.load("bvlc_googlenet.caffemodel")
m.load_label("labels.txt")
m.print_prediction("image.png")
