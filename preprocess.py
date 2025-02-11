import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

#import dataset 
fashion_train_dataset = pd.read_csv('dataset/fashion-mnist_train.csv')
fashion_test_dataset = pd.read_csv('dataset/fashion-mnist_test.csv')

#learn about the dataset
print(fashion_train_dataset.head())

# numpy array for the training and testing data
training = np.array(fashion_train_dataset, dtype = 'float32')
testing= np.array(fashion_test_dataset, dtype = 'float32')

print(training.shape)

#visualise an example image
i = random.randint(1,60000)
plt.imshow(training[i,1:].reshape(28,28))

#visualise a set of images with the respective label
W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid,W_grid, figsize = (17,17))
axes = axes.ravel() # flattens the 15 X 15 matrix into 225 array 

n_training = len(training)

for i in np.arange(0, W_grid * L_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(training[index,1:].reshape((28,28)))
    axes[i].set_title(f'Label: {int(training[index, 0])}', fontsize=8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()

# Fashion-MNIST Dataset Labels:
# 0 - T-shirt/top
# 1 - Trouser
# 2 - Pullover
# 3 - Dress
# 4 - Coat
# 5 - Sandal
# 6 - Shirt
# 7 - Sneaker
# 8 - Bag
# 9 - Ankle boot

 

