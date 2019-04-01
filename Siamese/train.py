from keras.optimizers import SGD,Adam
import os
from model_development.Siamese.model import get_siamese
from model_development.Siamese.Siamese_loader import Siamese_Loader

PATH = os.getcwd()
weights_path = os.path.join(PATH, "model_weights.h5")

model = get_siamese()
model.summary()

loader = Siamese_Loader(PATH)

evaluate_every = 10  # interval for evaluating on one-shot tasks
loss_every = 20  # interval for printing loss (iterations)
batch_size = 5
n_iter = 5000
N_way = 3  # how many classes for testing one-shot tasks>
n_val = 250  # how many one-shot tasks to validate on?

best = -1
print("Starting training process!")
print("-------------------------------------")
for i in range(1, n_iter):
    (inputs, targets) = loader.get_batch(batch_size)
    loss = model.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        val_acc = loader.test_oneshot(model, N_way, n_val, verbose=False)
        if val_acc >= best:
            print("Current best: {0}".format(val_acc))
            print("Saving weights to: {0} \n".format(PATH))
            model.save_weights(weights_path)
            best = val_acc

    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i, loss))
