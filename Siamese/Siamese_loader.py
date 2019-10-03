import numpy as np
import numpy.random as rng
import os
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, log_loss
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, path, feature_extractor, data_file='loader_data.pickle'):
        if os.path.isfile(data_file):
            self.data = pd.read_pickle(data_file)
        else:
            self.load_images(path, feature_extractor)
            self.data.to_pickle()

        self.classes = self.data.sock_name.unique()
        self.n_classes = len(self.classes)
        self.n_features = len(self.data.features[0])

    def get_batch(self, batch_size):
        """Create batch of n pairs, half same class, half different class"""
        pairs = np.zeros((batch_size, 2, self.n_features))
        # initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1

        for i in range(batch_size):
            if targets[i] == 1:
                # pick images of same class
                sock_name = self.data.sock_name.unique().sample(1)
                features = self.data[self.data.sock_name == sock_name].features.sample(2)
                features_1 = features[0]
                features_2 = features[1]
            else:
                # pick images of different classes
                sock_names = self.data.sock_name.unique().sample(2)
                features_1 = self.data[self.data.sock_name == sock_names[0]].features.sample(1)
                features_2 = self.data[self.data.sock_name == sock_names[1]].features.sample(1)
            pairs[i, 0] = features_1
            pairs[i, 1] = features_2
        return pairs, targets

    def generate(self, batch_size, s="train", replace=False):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size)
            yield (pairs, targets)

    def make_oneshot_task(self, N, s="val"):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        true_class = self.classes.sample(1)

        [test_features, true_features] = self.data[self.data.sock_name == true_class].features.sample(2)
        other_features = self.data[self.data.sock_name != true_class].features.sample(N-1)

        targets = np.zeros((N,))
        targets[0] = 1
        pairs = np.zeros((N, 2, self.n_features))
        pairs[:, 0, :] = test_features
        pairs[0, 1, :] = true_features
        pairs[1:, 1, :] = other_features

        return pairs, targets

    def test_oneshot(self, model, N, k, s="val", verbose=False):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N, s)
            probs = model.get_similarity(inputs)

            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
        return percent_correct

    def show_roc(self, model, batch_size):
        (inputs, targets) = self.get_batch(batch_size)
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
        (inputs, targets) = self.get_batch(batch_size)
        prediction = model.predict(inputs)
        y_true = targets
        y_pred = prediction.transpose()
        loss = log_loss(y_true, y_pred)
        return loss

    def load_images(self, path, feature_extractor):
        # loop through every sock, every image, and build a pandas DataFrame
        data = []
        for sock_dir in os.listdir(path):
            sock_path = os.path.abspath(os.path.join(path, sock_dir))
            for img_file in os.listdir(sock_path):
                img_path = os.path.join(sock_path, img_file)
                data.append({
                    'sock_name': sock_dir,
                    'img_path': img_path,
                })
        self.data = pd.DataFrame(data)
        self.data['features'] = [feature_extractor(img_path) for img_path in tqdm(self.data.img_path)]
