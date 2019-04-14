import os
import matplotlib.pyplot as plt
from bipedModel.model import get_siamese
from bipedResearch.Siamese.Siamese_loader import Siamese_Loader

PATH = os.getcwd()
weights_path = os.path.join(PATH, "../../bipedModel/model_weights.h5")

model = get_siamese()
model.summary()
#model.load_weights(weights_path)

loader = Siamese_Loader(PATH)

evaluate_every = 1000
batch_size = 14
epochs = 25
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
percent_correct = loader.test_oneshot(model, 5, 500)
print("Got an average of {}% one-shot learning accuracy \n".format(percent_correct))
loader.show_roc(model, 10000)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['val', 'test'], loc='upper left')
plt.show()
