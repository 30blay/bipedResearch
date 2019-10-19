from bipedResearch.Siamese.Siamese_loader import SiameseLoader
from bipedModel.model import BipedModel
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PATH = os.getcwd()

model = BipedModel()
loader = SiameseLoader(PATH + '/Train', model)

n_pairs = 100000

socks1 = loader.data.sample(n_pairs, replace=True)
socks2 = loader.data.sample(n_pairs, replace=True)

similarity = [model.get_similarity(f1, f2) for f1, f2 in zip(socks1.features, socks2.features)]
pairs = pd.DataFrame({
    'similarity': similarity,
    'sock1_class': socks1.sock_name.values,
    'sock2_class': socks2.sock_name.values,
    'sock1_path': socks1.img_path.values,
    'sock2_path': socks2.img_path.values,
})
pairs.sort_values('similarity', ascending=False, inplace=True)
pairs = pairs[pairs.sock1_class != pairs.sock2_class]

head = pairs.head(10)
for _, row in head.iterrows():
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(mpimg.imread(row['sock1_path']))
    axarr[0].title.set_text(row['sock1_class'])
    axarr[1].imshow(mpimg.imread(row['sock2_path']))
    axarr[1].title.set_text(row['sock2_class'])
    similarity = row['similarity']
    fig.suptitle(f'similarity = {similarity}')
    plt.show()
