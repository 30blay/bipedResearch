import os
import matplotlib.pyplot as plt
from bipedModel.model import get_siamese, weights_path
from bipedResearch.Siamese.Siamese_loader import Siamese_Loader

PATH = os.getcwd()

model = get_siamese(load_weights=False)
model.summary()

loader = Siamese_Loader(PATH)

evaluate_every = 1000
batch_size = 45
epochs = 20
N_way = 5  # how many classes for testing one-shot tasks

best = float("inf")
print("Starting training process!")
print("-------------------------------------")
history = model.fit_generator(loader.generate(batch_size),
                              epochs=epochs,
                              steps_per_epoch=evaluate_every,
                              validation_data=loader.generate(5, s='val'),
                              validation_steps=200,
                              )

model.save_weights(weights_path)
percent_correct = loader.test_oneshot(model, 5, 1000, verbose=False)
print("Got an average of {}% one-shot learning accuracy \n".format(percent_correct))
loader.show_roc(model, 10000)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
