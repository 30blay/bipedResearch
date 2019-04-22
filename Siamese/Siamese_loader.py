import numpy as np
import numpy.random as rng
import os
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy
from keras import backend as K


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, path, data_subsets=["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}

        for name in data_subsets:
            file_path = os.path.join(path, name + ".pickle")
            print("loading data from {}".format(file_path))
            with open(file_path, "rb") as f:
                (X, y, c) = pickle.load(f)
                self.data[name] = X
                self.categories[name] = c

    def get_batch(self, batch_size, s="train", replace=False):
        """Create batch of n pairs, half same class, half different class"""
        X = self.data[s]
        n_classes, n_examples, feature_len = X.shape

        # randomly sample several classes to use in the batch
        categories = rng.choice(n_classes, size=(batch_size,), replace=replace)
        # initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, feature_len)) for i in range(2)]
        # initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i, :] = X[category, idx_1].reshape(feature_len)
            idx_2 = rng.randint(0, n_examples)
            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category
            else:
                # add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1, n_classes)) % n_classes
            pairs[1][i, :] = X[category_2, idx_2].reshape(feature_len)
        return pairs, targets

    def generate(self, batch_size, s="train", replace=False):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size, s, replace=replace)
            yield (pairs, targets)

    def make_oneshot_task(self, N, s="val"):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X = self.data[s]
        n_classes, n_examples, w = X.shape
        indices = rng.randint(0, n_examples, size=(N,))
        categories = rng.choice(range(n_classes), size=(N,), replace=False)
        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
        test_image = np.asarray([X[true_category, ex1]] * N).reshape(N, w)
        support_set = X[categories, indices]
        support_set[0] = X[true_category, ex2]
        support_set = support_set.reshape(N, w)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]

        return pairs, targets, categories

    def test_oneshot(self, model, N, k, s="val", verbose=False):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
        for i in range(k):
            inputs, targets, categories = self.make_oneshot_task(N, s)
            probs = model.predict(inputs)

            if verbose:
                print("\nTrue category : {} ".format(self.categories[s][categories[np.argmax(targets)]]))
                for candidate, prob in zip(categories, probs):
                    print("{} : {}".format(self.categories[s][candidate], prob))

            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
        return percent_correct

    def show_roc(self, model, batch_size):
        (inputs, targets) = self.get_batch(batch_size, s='val', replace=True)
        prediction = model.predict(inputs)
        fpr, tpr, thresholds = roc_curve(targets, prediction, pos_label=1)
        fnr = np.ones(tpr.shape)-tpr
        plt.figure()
        plt.plot(thresholds, fnr, 'r', thresholds, fpr, 'g')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Threshold')
        plt.ylabel('Error rate')
        plt.title('Receiver operating characteristic')
        plt.legend(['Pair but false', 'Different but true'], loc="lower right")
        plt.show()

    def validate(self, model, batch_size):
        (inputs, targets) = self.get_batch(batch_size, s='val', replace=True)
        prediction = model.predict(inputs)
        y_true = K.variable(targets)
        y_pred = K.variable(prediction.transpose())
        loss = K.eval(binary_crossentropy(y_true, y_pred))
        return loss[0]

    def train(self, model, batch_size):
        (inputs, targets) = self.get_batch(batch_size)
        train_loss = model.train_on_batch(inputs, targets)
        return train_loss
