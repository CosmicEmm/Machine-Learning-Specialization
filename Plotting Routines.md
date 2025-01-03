```py
import numpy as np
import matplotlib.pyplot as plt


def plt_house_x(X, y, f_wb=None, ax=None):
    ''' plot house with axis '''
    if not ax: # If ax is not provided, create a new figure and Axes
        fig, ax = plt.subplots(1,1)

    ax.scatter(X, y, marker='x', c='r', label="Actual Value") # Plot data points

    ax.set_title("Housing Prices")  # Set the title of the plot
    ax.set_ylabel('Price (in 1000s of dollars)')  # Set the y-axis label
    ax.set_xlabel(f'Size (1000 sqft)')  # Set the x-axis label
    if f_wb is not None: # Plot prediction line if f_wb is provided
        ax.plot(X, f_wb, c='blue', label="Our Prediction")  
    ax.legend()  # Add a legend to the plot
```
#### Explanation
- If `ax` is not passed as an argument (`if not ax:`), a new [[Matplotlib#Subplots|figure and axes]] are created by calling `plt.subplots()`. This method returns a `figure` and an `axes` object.
- `ax.scatter()` creates a scatter plot of the data points (x, y) using red `x` markers.
- If `f_wb` is provided (not `None`), it will plot a line representing the predicted values. The line is drawn using `ax.plot()`.