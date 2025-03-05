import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])

# Select a linear model
model = KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(x, y)

# Plot the regression line
x_fit = np.linspace(23_500, 62_500, 1000).reshape(-1, 1)
y_fit = model.predict(x_fit)
plt.plot(x_fit, y_fit, color='red', linewidth=2, label='KNeighborsRegression Line')

# Save the plot to a file
plt.legend()
plt.savefig("graphs/lifesat_plot_with_KNeighborsRegression_line.png")  # Save the plot to a file
plt.close()  # Close the plot to free up memory

# Make a prediction for Cyprus
X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
print(model.predict(X_new))  # output: [[6.30165767]]