import pandas as pd
from haversine import haversine
import pulp
from pulp import GLPK, GUROBI
import folium

def read_data(file_path, nrows):
    df = pd.read_csv(file_path, nrows=nrows)
    places = df['Place_Name'].unique().tolist()
    coordinates = list(zip(df['Latitude'], df['Longitude']))
    return places, coordinates

def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = haversine(coordinates[i], coordinates[j])
            else:
                distance_matrix[i][j] = float('inf')  # Avoid self-loop by setting a high value
    return distance_matrix

def build_model(places, distance_matrix,):
    # instantiate the problem - Python PuLP model
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)

    # ***************************************************
    #   Defining decision variables
    # ***************************************************
    x = {}  # Binary: x_i,j:= 1 if visiting city j after city i; otherwise 0
    for i in places:
        for j in places:
            if distance_matrix[places.index(i)][places.index(j)] in sorted(distance_matrix[places.index(i)])[:10]:
                x[(i, j)] = pulp.LpVariable("x_" + str(places.index(i)) + '_' + str(places.index(j)), cat='Binary')

    s = {}  # Integer: s_i is the sequence number when visiting city i
    for i in places:
        s[i] = pulp.LpVariable("s_" + str(places.index(i)), cat='Integer', lowBound=0)

    # ********************************************
    # Objective
    # ********************************************
    # Minimize total travel distance
    obj_val = pulp.lpSum(x[(i, j)] * distance_matrix[places.index(i)][places.index(j)]
                         for i in places for j in places if i != j and (i, j) in x)
    prob += obj_val

    # ********************************************
    # Constraints
    # ********************************************
    # Each place should have exactly one outgoing connection
    for i in places:
        if any((i, j) in x for j in places if i != j):
            prob += pulp.lpSum(x[(i, j)] for j in places if i != j and (i, j) in x) == 1, 'Outgoing_sum_' + str(places.index(i))

    # Each place should have exactly one incoming connection
    for j in places:
        if any((i, j) in x for i in places if i != j):
            prob += pulp.lpSum(x[(i, j)] for i in places if i != j and (i, j) in x) == 1, 'Incoming_sum_' + str(places.index(j))

    # Sub-tour elimination constraint
    for i in places[1:]:
        for j in places[1:]:
            if i != j and (i, j) in x:
                prob += s[j] >= s[i] + 1 - len(places) * (1 - x[(i, j)]), 'sub_tour_' + str(places.index(i)) + '_' + str(places.index(j))

    return prob, x

def solve_tsp(prob, x, places):
    # Solve the problem
    solver = 'GUROBI'  # Solver choice: 'CBC', 'GUROBI', 'GLPK'
    print('-' * 50)
    print('Optimization solver', solver, 'called')
    print(f'TSP for {len(places)} cities. ')
    print('-' * 50)

    if solver == 'CBC':
        prob.solve(pulp.PULP_CBC_CMD(warmStart=True))
    elif solver == 'GUROBI':
        prob.solve(GUROBI(Heuristics=0.5, Cuts=2))
    else:
        print(solver, ' not available')
        exit()
    print(f'Status: {pulp.LpStatus[prob.status]}')

    if pulp.LpStatus[prob.status] == 'Optimal':
        n = len(places)
        optimal_route = [places[0]]
        current_place = places[0]
        while len(optimal_route) < n:
            for next_place in places:
                if current_place != next_place and (current_place, next_place) in x and x[(current_place, next_place)].value() > 0.5:
                    optimal_route.append(next_place)
                    current_place = next_place
                    break

        optimal_route.append(optimal_route[0])  # Return to the starting place

        total_distance = pulp.value(prob.objective)

#         print("Optimal Route:", " -> ".join(optimal_route))
        print("Total Distance:", total_distance)

        return optimal_route, total_distance
    else:
        print("No optimal solution found.")
        return None, None

# TSP
if __name__ == "__main__":
    data_file_path = 'Data/tsp_input.csv'
    places, coordinates = read_data(data_file_path, nrows=100)  # change number of places (nrows) according to model
    distance_matrix = calculate_distance_matrix(coordinates)
    problem, x = build_model(places, distance_matrix, )
    optimal_route, total_distance = solve_tsp(problem, x, places)
    print(f'Execution complete')
