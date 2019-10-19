import os
import shutil
from skimage import io
from bipedModel.model import BipedModel
import matplotlib.pyplot as plt


model = BipedModel()
data_dir = 'Siamese/Train'
output_dir = 'isolated'

shutil.rmtree(output_dir, ignore_errors=True)
os.mkdir(output_dir)

for sockdir in os.listdir(data_dir):
    sockpath = os.path.join(data_dir, sockdir)
    for imgfile in os.listdir(sockpath):
        imgpath = os.path.join(sockpath, imgfile)
        image = io.imread(imgpath)
        image = model.resize(image)
        image, mask = model.isolate_sock(image)
        plt.imshow(image)
        output_file = os.path.join(output_dir, imgpath.replace('/', '.'))
        plt.savefig(output_file)
