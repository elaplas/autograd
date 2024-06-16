import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from var import Var



# Set the random seed for reproducibility (optional)
np.random.seed(42)

# Generate 20 random 2D points in the range (-20, 20)
points = np.random.uniform(-20, 20, (20, 2))

# Classify the points into two classes
# Here, we simply classify based on the x-coordinate for demonstration purposes
class_labels = (points[:, 0] > 0).astype(float)
class_labels[class_labels==0]=-1.0

data_points = [list(points[i]) for i in range(len(points))]
classes = [[class_labels[i]] for i in range(len(class_labels))]


nn = MLP([2,4,4,2,1])
learning_rate = 0.02
for s in range(100):
    loss = Var(0.0, "loss")
    for i in range(len(data_points)):
        y_i = nn(data_points[i])
        for j in range(len(y_i)):
            loss += (y_i[j]-classes[i][j])**2
    loss.backward()
    W = nn.weights()
    for k in range(len(W)):
        W[k].data = W[k].data - (learning_rate * W[k].grad)
        W[k].grad = 0.0
    print(f"loss {s}: {loss}")


# Plot the points
colors = {-1:'red', 1:'blue'}
for class_label in np.unique(class_labels):
    class_points = points[class_labels == class_label]
    plt.scatter(class_points[:, 0], class_points[:, 1], color=colors[class_label], label=f'Class {class_label}')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Points Classification')
plt.legend()
plt.grid(True)
plt.show()

print(nn([-10, 10]))
print(nn([10, 10]))
