import sys
import numpy as np
from scipy.misc import imread
import pickle
import os
import matplotlib.pyplot as plt
import argparse
from keras.applications.vgg16 import VGG16
from keras import models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from bipedResearch.Siamese.model import get_encoder

"""Script to preprocess the sock dataset and pickle it into an array"""

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Path where omniglot folder resides")
parser.add_argument("--save", help="Path to pickle data to.", default=os.getcwd())
args = parser.parse_args()
train_folder = os.path.join(args.path, 'training')
val_folder = os.path.join(args.path, 'validation')

save_path = args.save


def loadimgs(path):

    X = []
    y = []
    cat_dict = {}
    curr_y = 0
    model = get_encoder()
    max_examples_per_cat = 6

    # every sock has it's own column in the array, so load seperately
    for sockCat in os.listdir(path):
        cat_dict[curr_y] = sockCat
        print(sockCat)
        sock_path = os.path.join(path, sockCat)
        category_features = []

        datagen = ImageDataGenerator(rescale=1. / 255)
        for filename in os.listdir(sock_path):
            image_path = os.path.join(sock_path, filename)
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            generator = datagen.flow(x)
            for inputs_batch in generator:
                features = model.predict(inputs_batch)
                break
            category_features.append(features[0])
            if len(category_features) >= max_examples_per_cat:
                break
        y.append(curr_y)
        try:
            X.append(np.stack(category_features))
        # edge case  - last one
        except ValueError as e:
            print(e)
        curr_y += 1
    y = np.vstack(y)
    X = np.stack(X)
    return X, y, cat_dict


X, y, cat_dict = loadimgs(train_folder)
with open(os.path.join(save_path, "train.pickle"), "wb") as f:
    pickle.dump((X, y, cat_dict), f)

X, y, cat_dict = loadimgs(val_folder)
with open(os.path.join(save_path, "val.pickle"), "wb") as f:
    pickle.dump((X, y, cat_dict), f)
