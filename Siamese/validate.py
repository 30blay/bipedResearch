import os
import matplotlib.pyplot as plt
from bipedModel.model import BipedModel
from bipedResearch.Siamese.Siamese_loader import SiameseLoader

PATH = os.getcwd()

model = BipedModel()

loader = SiameseLoader(PATH + '/Train', model)

evaluate_every = 1000
batch_size = 45
epochs = 10
N_way = 5  # how many classes for testing one-shot tasks

print("Starting validation process!")
print("-------------------------------------")

percent_correct = loader.test_oneshot(model, 5, 10000, verbose=False)
print("Got an average of {}% one-shot learning accuracy \n".format(percent_correct))
loader.show_roc(model, 10000)
