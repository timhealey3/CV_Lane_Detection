import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
np.random.seed(0)
# number of points
n_pts = 500
# data points as x, labels in y
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

#plt.scatter(X[y==0, 0], X[y==0, 1])
#plt.scatter(X[y==1,0], X[y==1,1])
#plt.show()

# add input layers
model = Sequential()
# hidden layers
# 4 neurons on hidden layer, 2 neurons for input
model.add(Dense(4, input_shape=(2,), activation='sigmoid'))
# add output layers
model.add(Dense(1, activation='sigmoid'))
# compile model with adam
model.compile(Adam(learning_rate=0.01), 'binary_crossentropy', metrics=['accuracy'])
# model that best classifies
h = model.fit(x=X, y=y, verbose = 1, batch_size=20, epochs=100, shuffle='true')

plt.plot(h.history['accuracy'])
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.show()

plt.plot(h.history['loss'])
plt.xlabel('epoch')
plt.legend(['loss'])
plt.show()

def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 0.25, max(X[:,0]) + 0.25)
    y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:,1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

plot_decision_boundary(X, y, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])

plot_decision_boundary(X, y, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])

x = 0
y = 0.75

point = np.array([[x, y]])
predict = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
print("Prediction is: ", predict)
