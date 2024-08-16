# Traveling Salesman Problem (TSP) Optimization

## Overview

The Traveling Salesman Problem (TSP) is a classic optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the starting point. This project explores solving the TSP using parameter tuning, warm start techniques, and heuristics to improve route optimization. It integrates business intelligence methods to prioritize nearby cities, aiming to enhance the efficiency of the solution.

## Features

- **Data Handling:** Reads city data from CSV files.
- **Distance Calculation:** Computes distances using the Haversine formula.
- **Model Building:** Constructs a TSP model using PuLP with integer programming.
- **Heuristics & Warm Start:** Utilizes warm start solution genereted using Google OR tools and heuristics to improve the solution process.
- **Solver Integration:** Solves the problem using Gurobi or CBC solvers.
- **Route Visualization:** Plots the optimal route on an interactive map using Folium.

## Code Explanation

- **`read_data(file_path, nrows)`:** Reads the city data from a CSV file and extracts place names and coordinates.
- **`calculate_distance_matrix(coordinates)`:** Computes the distance matrix using the Haversine formula.
- **`build_model(places, distance_matrix, warm_start_df)`:** Constructs the TSP model with constraints and objective function.
- **`solve_tsp(prob, x, places)`:** Solves the TSP using the specified solver and extracts the optimal route.
- **`plot_route(optimal_route, coordinates, places)`:** Visualizes the optimal route on an interactive map using Folium.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## Acknowledgments

- **PuLP**: [PuLP Documentation](https://coin-or.github.io/pulp/)
- **Gurobi**: [Gurobi Documentation](https://www.gurobi.com/documentation/9.5/refman/)
- **Folium**: [Folium Documentation](https://python-visualization.github.io/folium/)
- **OR tools**: [Google OR tools](https://developers.google.com/optimization)
