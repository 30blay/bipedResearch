import numpy as np
from keras import models
from keras.applications.vgg16 import VGG16


def extract_features(directory, nPerClass, datagen, nClasses, conv_base, augmented_dir=None):
    features = np.zeros(shape=(nClasses * nPerClass, 4096))
    labels = np.zeros(shape=(nClasses * nPerClass, nClasses))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        save_to_dir=augmented_dir)
    i = 0

    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)

        features[i] = features_batch
        labels[i] = labels_batch
        i += 1
        if i >= nClasses*nPerClass:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


def preprocess(train_dir, train_datagen, validation_dir, val_datagen, train_fea_path, valid_fea_path, augment_factor, nClasses, train_lab_path, valid_lab_path, augmented_dir):
    vgg16_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    conv_base = models.Sequential()
    for layer in vgg16_model.layers[:-1]:  # just exclude last layer from copying
        conv_base.add(layer)

    conv_base.summary()

    train_features, train_labels = extract_features(train_dir, 2*augment_factor, train_datagen, nClasses, conv_base, augmented_dir=augmented_dir)
    validation_features, validation_labels = extract_features(validation_dir, 1, val_datagen, nClasses, conv_base)

    open(train_fea_path, 'w')
    open(train_lab_path, 'w')
    open(valid_fea_path, 'w')
    open(valid_lab_path, 'w')

    np.save(train_fea_path, train_features)
    np.save(valid_fea_path, validation_features)
    np.save(train_lab_path, train_labels)
    np.save(valid_lab_path, validation_labels)
