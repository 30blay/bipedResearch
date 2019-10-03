import os
import matplotlib.pyplot as plt
from bipedModel.model import BipedModel
from bipedResearch.Siamese.Siamese_loader import Siamese_Loader

PATH = os.getcwd()

model = BipedModel()

loader = Siamese_Loader(PATH+'/Train', model.extract_features)

evaluate_every = 1000
batch_size = 45
epochs = 10
N_way = 5  # how many classes for testing one-shot tasks

best = float("inf")
print("Starting validation process!")
print("-------------------------------------")

percent_correct = loader.test_oneshot(model, 5, 1000, verbose=False)
print("Got an average of {}% one-shot learning accuracy \n".format(percent_correct))
loader.show_roc(model, 10000)
