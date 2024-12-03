import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize
import math


if __name__ == '__main__':
    size_of_bounds = 8 # Change to how much car length you want

    df = pd.read_csv('~/Downloads/BrandsHatchLayout.csv') # Change to where the file is
    x_column = df.iloc[1:, 0].to_numpy()
    y_column = df.iloc[1:, 1].to_numpy()
    x_column = x_column
    y_column = y_column

    x_inner = np.array(x_column[157:])
    x_outer = np.array(x_column[:156])
    y_inner = np.array(y_column[157:])
    y_outer = np.array(y_column[:156])

    x_inner_limit = []
    x_outer_limit = []
    y_inner_limit = []
    y_outer_limit = []
    for i in range(len(x_inner)):
        direction = [(x_outer[i] - x_inner[i]), (y_outer[i] - y_inner[i])]
        size = math.sqrt(direction[0]**2+direction[1]**2)
        x_inner_limit.append(x_inner[i]+direction[0]*size_of_bounds/size)
        x_outer_limit.append(x_outer[i]-direction[0]*size_of_bounds/size)
        y_inner_limit.append(y_inner[i]+direction[1]*size_of_bounds/size)
        y_outer_limit.append(y_outer[i]-direction[1]*size_of_bounds/size)

    middle_line_x = []
    middle_line_y = []
    for i in range(int(len(x_column)/2)):
        middle_line_x.append((x_column[i]+x_column[i+1+int(len(x_column)/2)])/2)
        middle_line_y.append((y_column[i]+y_column[i+1+int(len(y_column)/2)])/2)
    t = np.linspace(0, len(middle_line_x)-1, len(middle_line_x))
    spline_x_inner = make_interp_spline(t, x_inner_limit, k=5)
    spline_x_outer = make_interp_spline(t, x_outer_limit, k=5)
    spline_y_inner = make_interp_spline(t, y_inner_limit, k=5)
    spline_y_outer = make_interp_spline(t, y_outer_limit, k=5)
    spline_x_middle = make_interp_spline(t, middle_line_x, k=5)
    spline_y_middle = make_interp_spline(t, middle_line_y, k=5)

    num_dense_points = 500
    t_dense = np.linspace(0, len(middle_line_x) - 1, num_dense_points)  # More points for optimization
    dense_x = spline_x_middle(t_dense)
    dense_y = spline_y_middle(t_dense)
    dense_x_outer = spline_x_outer(t_dense)
    dense_y_outer = spline_y_outer(t_dense)
    dense_x_inner = spline_x_inner(t_dense)
    dense_y_inner = spline_y_inner(t_dense)

    def calculate_total_length(points):
        n_points = len(points) // 2
        x, y = points[:n_points], points[n_points:]
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        return np.sum(distances)
    bounds = []
    for i in range(len(dense_x)):
        bounds.append((min(dense_x_inner[i], dense_x_outer[i]),
                       max(dense_x_inner[i], dense_x_outer[i])))
    for i in range(len(dense_x)):
        bounds.append((min(dense_y_inner[i], dense_y_outer[i]),
                       max(dense_y_inner[i], dense_y_outer[i])))

    initial_guess = np.concatenate([dense_x, dense_y])
    result = minimize(calculate_total_length, initial_guess, bounds=bounds, method='L-BFGS-B',
                      options={'disp': True, 'maxiter': 500, 'gtol': 1e-4})
    optimized_x = result.x[:len(result.x) // 2]
    optimized_y = result.x[len(result.x) // 2:]

    plt.figure(figsize=(10, 10))

    plt.plot(dense_x, dense_y, label=' Middle Line', color='blue')
    plt.plot(result.x[:len(result.x) // 2], result.x[len(result.x) // 2:], label='Optimized Middle Line', color='red')

    plt.plot(dense_x_outer, dense_y_outer, label='Inner Limit', color='black')
    plt.plot(dense_x_inner, dense_y_inner, label='Outer Limit', color='black')
    plt.scatter(x_column, y_column, marker='o')
    plt.legend()
    plt.title('Initial vs Optimized Middle Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
