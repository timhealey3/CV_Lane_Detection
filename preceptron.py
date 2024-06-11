import numpy as np 
import tensorflow.keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt 

n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
                   np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
                   np.random.normal(6, 2, n_pts)]).T
     
X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T
     
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

model = Sequential()
model.add(Dense(units = 1, input_shape=(2,), activation='sigmoid'))
adam = Adam(learning_rate = 0.1)
# use adam and binary cross entropy. for two different classes
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
# label model
# shuffle, shuffles the data before each epoch to help fight against gettincaught in local min.
h = model.fit(x=X, y=y, verbose=1, batch_size=50, epochs = 500, shuffle='true')

