import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib.colors import ListedColormap
#DONT USE THIS VERSION, GO TO JUPYTER NOTEBOOK ONE INSTEAD
# parameters for Gaussian clusters
cov = [[.5, 0], [0, .5]]  # covariance matrix
mean1 = [2, 2] 
mean2 = [5, 2]
mean3 = [4, 6]
size = 300      # points/cluster

# Generate the data for each cluster
cluster1 = np.random.multivariate_normal(mean1, cov, size)
cluster2 = np.random.multivariate_normal(mean2, cov, size)
cluster3 = np.random.multivariate_normal(mean3, cov, size)



#train the model
model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),
    Dense(3, activation='softmax')  # Output layer for 3-class classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

zeros = np.zeros((cluster1.shape[0], 1))
cluster1wanswer = np.hstack((cluster1, zeros))

ones = np.full((cluster2.shape[0], 1), 1)
cluster2wanswer = np.hstack((cluster2, ones))

twos = np.full((cluster3.shape[0], 1), 2)
cluster3wanswer = np.hstack((cluster3, twos))
everything = np.concatenate((cluster1wanswer, cluster2wanswer, cluster3wanswer))
np.random.shuffle(everything)
X = everything[:, 0:2]
y = everything[:, 2]

model.fit(X, y, epochs=50, batch_size=10)



#graphing prediction boundaries
# Step 1: Create a meshgrid over the input space
h = 0.05  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Step 2: Predict the class for each point in the grid
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

# Step 3: Plot the decision boundary
custom_cmap = ListedColormap(["purple", "blue", "pink"])
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=custom_cmap, alpha=0.5)

# Overlay the original training data
plt.scatter(cluster1[:,0], cluster1[:,1], color="purple", label="cluster 1")
plt.scatter(cluster2[:,0], cluster2[:,1], color="blue", label="cluster 2")
plt.scatter(cluster3[:,0], cluster3[:,1], color="pink", label="cluster 3")
plt.title("Decision Boundaries of MLP Classifier")

weights1, biases1 = model.layers[0].get_weights()
print("Weights for first layer:\n", weights1)
print("Biases for first layer:\n", biases1)

weights2, biases2 = model.layers[1].get_weights()
print("Weights for second layer:\n", weights2)
print("Biases for second layer:\n", biases2)

loss, accuracy = model.evaluate(X, y, verbose=0)
print("Accuracy:", accuracy)
plt.show()

