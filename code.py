import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create simple dataset (hours studied vs marks)
hours = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
marks = np.array([35, 40, 50, 55, 65, 70])

# Create and train model
model = LinearRegression()
model.fit(hours, marks)

# Predict values
predicted_marks = model.predict(hours)

# Plot results
plt.figure(figsize=(6,4))
plt.scatter(hours, marks)
plt.plot(hours, predicted_marks)
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple ML Model: Study Hours vs Marks")
plt.show()