import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
n_pts = 500
centers = [[-1, 1], [-1, -1], [1, -1]]
# data points X, and labels y
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123,
    centers=centers, cluster_std=0.4)
plt.scatter(X[y==0, 0], X[y==0, 1])
# one hot encoding
y_cat = to_categorical(y, 3)
# make perceptron NN
model = Sequential()
model.add(Dense(units=3, input_shape=(2,),activation='softmax'))
model.compile(Adam(0.1), loss='categorical_crossentropy',
    metrics=['accuracy'])
# train data
model.fit(x=X, y=y_cat, verbose=1, batch_size=50, epochs=100)

# plotting
# plot decision boundary here
