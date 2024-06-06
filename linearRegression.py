# logistic regression
import matplotlib.pyplot as plt
import numpy as np 

def draw(x1, x2):
    ln = plt.plot(x1, x2)
def sigmoid(score):
    return 1 / (1 + np.exp(-score))
# generate random points that can be split into top and bot
np.random.seed(0)
n_pts = 100
# bias for perceptron
bias = np.ones(n_pts)
# 10 is center, n_pts amount of points, std 2
random_x1_values = np.random.normal(10, 2, n_pts)
random_x2_values = np.random.normal(12,2,n_pts)
top_region = np.array([random_x1_values, random_x2_values, bias]).T 
bottom_region = np.array([np.random.normal(5,2,n_pts), np.random.normal(6,2,n_pts), bias]).T
# combine into one 
all_points = np.vstack((top_region, bottom_region))
# make linear linear with some random weights and bias
w1 = -.2 
w2 = -.35 
bias = 3.5
line_parameters = np.matrix([w1, w2, bias]).T
# find x and y coordinates
x1 = np.array([bottom_region[:,0].min(), top_region[:,0].max()])
x2 = -bias / w2 + x1 + (-w1 / w2)
# all points matrix multiplaction with (weights and bias)
linear_combination = all_points * line_parameters
probabilities = sigmoid(linear_combination)

# graphing w/ matplotlib
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:, 0], top_region[:, 1], color="red")
ax.scatter(bottom_region[:,0], top_region[:, 1], color="blue")
draw(x1, x2)
plt.show()
