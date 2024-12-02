import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize
from scipy.integrate import quad
import math


if __name__ == '__main__':

    df = pd.read_csv('~/Downloads/BrandsHatchLayout.csv')
    x_column = df.iloc[1:, 0].to_numpy()
    y_column = df.iloc[1:, 1].to_numpy()
    x_column = x_column
    y_column = y_column

    x_inner = np.array(x_column[157:])
    x_outer = np.array(x_column[:156])
    y_inner = np.array(y_column[157:])
    y_outer = np.array(y_column[:156])

    # x_inner_limit = (x_inner+x_outer)/2.1
    # x_outer_limit = (x_outer+x_inner)/1.9
    # y_inner_limit = (y_inner+y_outer)/2.1
    # y_outer_limit = (y_outer+y_inner)/1.9
    # x_inner_limit = (x_inner)/0.95
    # x_outer_limit = (x_outer)/1.05
    # y_inner_limit = (y_inner)/0.95
    # y_outer_limit = (y_outer)/1.05
    # Display the first few rows of the data
    # print(df)
    # print("------------------")
    # print(df.head())
    # print(y_column)
    x_inner_limit = []
    x_outer_limit = []
    y_inner_limit = []
    y_outer_limit = []
    for i in range(len(x_inner)):
        direction = [(x_outer[i] - x_inner[i]), (y_outer[i] - y_inner[i])]
        size = math.sqrt(direction[0]**2+direction[1]**2)
        x_inner_limit.append(x_inner[i]+direction[0]*5/size)
        x_outer_limit.append(x_outer[i]-direction[0]*5/size)
        y_inner_limit.append(y_inner[i]+direction[1]*5/size)
        y_outer_limit.append(y_outer[i]-direction[1]*5/size)

    middle_line_x = []
    middle_line_y = []
    for i in range(int(len(x_column)/2)):
        middle_line_x.append((x_column[i]+x_column[i+1+int(len(x_column)/2)])/2)
        middle_line_y.append((y_column[i]+y_column[i+1+int(len(y_column)/2)])/2)
    points = np.column_stack((middle_line_x, middle_line_y))
    t = np.linspace(0, len(middle_line_x), len(middle_line_x))
    spline_x_inner = make_interp_spline(t, x_inner_limit, k=3)
    spline_x_outer = make_interp_spline(t, x_outer_limit, k=3)
    spline_y_inner = make_interp_spline(t, y_inner_limit, k=3)
    spline_y_outer = make_interp_spline(t, y_outer_limit, k=3)
    spline_x_middle = make_interp_spline(t, middle_line_x, k=3)
    spline_y_middle = make_interp_spline(t, middle_line_y, k=3)
    t_dense = np.linspace(0, 155, 2000)  # More points for optimization
    dense_x = spline_x_middle(t_dense)
    dense_y = spline_y_middle(t_dense)
    dense_x_outer = spline_x_outer(t_dense)
    dense_y_outer = spline_y_outer(t_dense)
    dense_x_inner = spline_x_inner(t_dense)
    dense_y_inner = spline_y_inner(t_dense)
    def calculate_total_curvature(points):
        x, y = points[:len(points) // 2], points[len(points) // 2:]
        spline_x = make_interp_spline(t, middle_line_x, k=3)
        spline_y = make_interp_spline(t, middle_line_y, k=3)
        dx_dt = spline_x.derivative(1)
        dy_dt = spline_y.derivative(1)
        ddx_dt = spline_x.derivative(2)
        ddy_dt = spline_y.derivative(2)
        def curvature(tt):
            numerator = np.abs(dx_dt(tt) * ddy_dt(tt) - dy_dt(tt) * ddx_dt(tt))
            denominator = (dx_dt(tt) ** 2 + dy_dt(tt) ** 2) ** 1.5
            return numerator / denominator
        # Integrate curvature over the entire path
        curvature_values = [curvature(ti) for ti in t]
        return np.sum(curvature_values)

    bounds = []
    for i in range(len(middle_line_x)):
        bounds.append((min(x_column[i], x_column[i + 1 + int(len(x_column) / 2)]),
                       max(x_column[i], x_column[i + 1 + int(len(x_column) / 2)])))
    for i in range(len(middle_line_y)):
        bounds.append((min(y_column[i], y_column[i + 1 + int(len(y_column) / 2)]),
                       max(y_column[i], y_column[i + 1 + int(len(y_column) / 2)])))

    initial_guess = np.concatenate([middle_line_x, middle_line_y])
    result = minimize(calculate_total_curvature, initial_guess, bounds=bounds, method='L-BFGS-B')
    optimized_x = result.x[:len(result.x) // 2]
    optimized_y = result.x[len(result.x) // 2:]

    # t_smooth = np.linspace(0, len(middle_line_x), 2000)
    # x_smooth = spline_x(t_smooth)
    # y_smooth = spline_y(t_smooth)
    # x_smooth = spline_x(t_smooth)
    # y_smooth = spline_y(t_smooth)
    # x_smooth = np.linspace(np.array(middle_line_x).min(), np.array(middle_line_x).max(), 500)
    # spline = make_interp_spline(middle_line_x, middle_line_y, k=3)
    # y_smooth = spline(x_smooth)
    # print(f"x_smooth: {spline_x}")
    # print(f"y_smooth: {spline_y}")
    # print(f"x_smooth: {x_smooth}")
    # print(f"y_smooth: {y_smooth}")

    # plt.figure(figsize=(10, 10))  # Optional: set figure size
    # plt.scatter(x_column, y_column, marker='o')  # Scatter plot
    # #plt.scatter(middle_line_x, middle_line_y, marker='o')
    # plt.plot(optimized_x, optimized_y, label='Optimized Middle Line', color='red')
    # # Add labels and title
    # plt.xlabel('X Values')
    # plt.ylabel('Y Values')
    # plt.title('Scatter Plot of Dots')
    #
    # # Display the plot
    # plt.grid(True)  # Optional: adds gridlines
    # plt.show()
    # plt.close()

    plt.figure(figsize=(10, 10))  # Optional: set figure size
    plt.scatter(x_column, y_column, marker='o')  # Scatter plot
    #plt.scatter(middle_line_x, middle_line_y, marker='o')
    plt.plot(middle_line_x, middle_line_y, label='Smooth Path', color='blue')
    # plt.plot(dense_x_inner, dense_y_inner, label='Inner Limit', color='black')
    # plt.plot(dense_x_outer, dense_y_outer, label='Outer Limit', color='black')
    plt.scatter(x_inner_limit, y_inner_limit, label='Inner Limit', color='black')
    plt.scatter(x_outer_limit, y_outer_limit, label='Outer Limit', color='black')
    # Add labels and title
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.title('Scatter Plot of Dots')

    # Display the plot
    plt.grid(True)  # Optional: adds gridlines
    plt.show()