from keras.optimizers import SGD,Adam
import os
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
n_iter = 100000
N_way = 5  # how many classes for testing one-shot tasks
n_val = 200  # how many one-shot tasks to validate on?

best = float("inf")
print("Starting training process!")
print("-------------------------------------")
for i in range(1, n_iter):
    (inputs, targets) = loader.get_batch(batch_size)
    loss = model.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        val_acc = loader.test_oneshot(model, N_way, n_val, verbose=False)
        val_loss = loader.validate(model, 1000)
        if val_loss <= best:
            print("Current best: {0}".format(val_loss))
            print("Saving weights to: {0} \n".format(PATH))
            best = val_loss
        print("iteration {}, training loss: {:.2f}, val loss: {:.2f}, val accuracy: {:.2f}".format(i, loss, val_loss, val_acc))

model.save_weights(weights_path)
loader.show_roc(model, 10000)
