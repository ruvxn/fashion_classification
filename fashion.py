import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tensorflow.keras import datasets, layers, models

#import dataset 
fashion_train_dataset = pd.read_csv('dataset/fashion-mnist_train.csv')
fashion_test_dataset = pd.read_csv('dataset/fashion-mnist_test.csv')


# numpy array for the training and testing data
training = np.array('fashion_train_dataset', dtype = 'float32')
testing = np.array('fashion_test_dataset', dtype = 'float32')

#normalising data
X_train = training[:,1:]/255
y_train = training[:, 0]

X_test = testing[:,1:]/255
y_train = testing[:, 0]

X_train = X_train.reshape(X_train.shape[0], *(28,28,1))
X_test = X_test.reshape(X_test.shape[0], *(28,28,1))

#build the model

cnn = models.Sequential()
cnn.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
cnn.add(layers.MaxPooling2D(2,2))

cnn.add(layers.Conv2D(64,(3,3), activation='relu'))
cnn.add(layers.MaxPooling2D(2,2))

cnn.add(layers.Conv2D(64,(3,3), activation='relu'))
cnn.add(layers.Flatten())

cnn.add(layers.Conv2D(64, activation='relu'))
cnn.add(layers.Conv2D(10, activation='softmax'))

#compile mode;

cnn.compile(loss='sparse categorical crossentropy', optimizer='Adam(learning_rate=0.001)', metrics = 'accuracy')

#train model
epochs = 25
history = cnn.fit(X_train, y_train, batch_size = 256, epochs = epochs)

#evaluate model
test_loss, test_acc = cnn.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

# Pick a random test image
index = random.randint(0, len(X_test) - 1)
image = X_test[index]

# Make a prediction
prediction = cnn.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(prediction)

# Display Image
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.show()