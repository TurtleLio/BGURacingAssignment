import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('~/Downloads/BrandsHatchLayout.csv')
    x_column = df.iloc[1:, 0].to_numpy()
    y_column = df.iloc[1:, 1].to_numpy()

    middle_line_x = []
    middle_line_y = []
    for i in range(int(len(x_column)/2)):
        middle_line_x.append((x_column[i]+x_column[i+1+int(len(x_column)/2)])/2)
        middle_line_y.append((y_column[i]+y_column[i+1+int(len(y_column)/2)])/2)

    t = np.linspace(0, len(middle_line_x), len(middle_line_x))
    spline_x = make_interp_spline(t, middle_line_x, k=3)
    spline_y = make_interp_spline(t, middle_line_y, k=3)
    t_smooth = np.linspace(0, len(middle_line_x), 2000)
    x_smooth = spline_x(t_smooth)
    y_smooth = spline_y(t_smooth)

    plt.figure(figsize=(10, 10))  # Optional: set figure size
    plt.scatter(x_column, y_column, marker='o')  # Scatter plot
    #plt.scatter(middle_line_x, middle_line_y, marker='o')
    plt.plot(x_smooth, y_smooth, label='Smooth Path', color='blue')
    # Add labels and title
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.title('Scatter Plot of Dots')

    # Display the plot
    plt.grid(True)  # Optional: adds gridlines
    plt.show()

