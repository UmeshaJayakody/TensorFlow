# Import necessary libraries for plotting and numerical operations
import matplotlib.pyplot as plt
import numpy as np

# Define sample data points for x and y coordinates
x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]

# Create and save a basic scatter plot
plt.plot(x, y, 'ro')  # Plot red circles at each (x,y) point
plt.axis([0, 6, 0, 20])  # Set x-axis from 0-6, y-axis from 0-20
plt.savefig('./graph_output/module_03/plot_1.png')  # Save the scatter plot as PNG

# Create and save a scatter plot with linear regression line
plt.plot(x, y, 'ro')  # Plot the same red circles
plt.axis([0, 6, 0, 20])  # Set axis limits again
# Fit a linear polynomial to the data and plot the regression line
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.savefig('./graph_output/module_03/plot_2.png')  # Save the plot with line as PNG