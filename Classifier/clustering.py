import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from preprocess import preprocess
from SphericalClassifier import SphericalClassifier
from sklearn.manifold import TSNE
from numpy import argmax

augment_factor = 10
nClasses = 12
nEpochs = 1600
save_augmented = False
enable_preprocess = True

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=0,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.1,
      horizontal_flip=False,
      fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

base_dir = dir_path = os.path.dirname(os.path.realpath(__file__))
train_dir = os.path.join(base_dir, '../Train')
validation_dir = os.path.join(base_dir, '../Validation')
augmented_dir = os.path.join(base_dir, '../Augmented')
train_fea_path = os.path.join(base_dir, 'train_fea.npy')
train_lab_path = os.path.join(base_dir, 'train_lab.npy')
valid_fea_path = os.path.join(base_dir, 'valid_fea.npy')
valid_lab_path = os.path.join(base_dir, 'valid_lab.npy')
if not save_augmented:
    augmented_dir = ""

if enable_preprocess:
    preprocess(
        train_dir,
        train_datagen,
        validation_dir,
        val_datagen,
        train_fea_path,
        valid_fea_path,
        augment_factor,
        nClasses,
        train_lab_path,
        valid_lab_path,
        augmented_dir
    )

train_features = np.load(train_fea_path)
validation_features = np.load(valid_fea_path)
train_labels = np.load(train_lab_path)
validation_labels = np.load(valid_lab_path)

model = models.Sequential()
#model.add(layers.Dense(nClasses, activation='softmax', input_dim=4096))
model.add(SphericalClassifier(nClasses))


#model.summary()

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

#model.load_weights('trained_weights.h5')

history = model.fit(train_features, train_labels,
                    epochs=nEpochs,
                    validation_data=(validation_features, validation_labels))


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()

weights = model.get_weights()

model.save_weights('trained_weights.h5')

X_embedded = TSNE(n_components=2).fit_transform(train_features)

for i in range(nClasses):
    classes = [argmax(label) for label in train_labels]
    idx = [category == i for category in classes]
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=i)

plt.legend(bbox_to_anchor=(1, 1))
plt.show()
