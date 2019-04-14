from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import models
import os.path
from bipedResearch.Siamese.model import get_encoder, get_siamese

PATH = os.getcwd()
test_path = os.path.join(PATH, "/Test")

encoder = get_encoder()
siamese = get_siamese()

val_datagen = ImageDataGenerator(rescale=1. / 255)
all_features = []

for filename in os.listdir(test_path):
    img = image.load_img(test_path + filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    generator = val_datagen.flow(x)

    for inputs_batch in generator:
        features = encoder.predict(inputs_batch)
        all_features.append(features[0])
        break


def get_similarity(feature1, feature2):
    input_tensor = np.vstack((feature1, feature2))

    siamese = get_siamese()
    APP_ROOT = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(APP_ROOT, "model_weights.h5")
    siamese.load_weights(weights_path)
    similarity = siamese.predict([feature1, feature2])
    return similarity[0][0]