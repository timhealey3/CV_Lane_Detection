# logistic regression
import matplotlib.pyplot as plt
import numpy as np 
np.random.seed(0)
n_pts = 100
# 10 is center, n_pts amount of points, std 2
random_x1_values = np.random.normal(10, 2, n_pts)
random_x2_values = np.random.normal(12,2,n_pts)
top_region = np.array([random_x1_values, random_x2_values]).T 
bottom_region = np.array([np.random.normal(5,2,n_pts), np.random.normal(6,2,n_pts)]).T
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:, 0], top_region[:, 1], color="red")
ax.scatter(bottom_region[:,0], top_region[:, 1], color="blue")
plt.show()
