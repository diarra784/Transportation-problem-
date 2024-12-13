import pandas as pd
import numpy as np

# Step 1: Load Parameters from External File
def load_parameters(file_path):
    data = pd.read_csv("C:/Users/DELL/Downloads/,Supplier1,Supplier2,Supplier3,Supp.txt")
    data = data.apply(pd.to_numeric, errors='coerce')
    print(data)  
    return data

# Step 2: Northwest Corner Rule
def northwest_corner(supply, demand):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    i, j = 0, 0

    while i < m and j < n:
        allocation[i][j] = min(supply[i], demand[j])
        supply[i] -= allocation[i][j]
        demand[j] -= allocation[i][j]

        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1

    return allocation

# Step 3: Minimum Cost Method
def minimum_cost_method(costs, supply, demand):
    m, n = costs.shape
    allocation = np.zeros((m, n))
    
    while np.sum(supply) > 0 and np.sum(demand) > 0:
        min_cost_index = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
        i, j = min_cost_index

        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]
        
        if supply[i] == 0:
            costs[i, :] = np.inf
        if demand[j] == 0:
            costs[:, j] = np.inf

    return allocation

# Step 4: Minimum Row Cost Method
def minimum_row_cost_method(costs, supply, demand):
    m, n = costs.shape
    allocation = np.zeros((m, n))

    for i in range(m):
        while supply[i] > 0:
            j = np.argmin(costs[i])
            allocation[i, j] = min(supply[i], demand[j])
            supply[i] -= allocation[i, j]
            demand[j] -= allocation[i, j]
            
            if demand[j] == 0:
                costs[:, j] = np.inf

    return allocation

# Step 5: Vogel's Approximation Method
def vogels_method(costs, supply, demand):
    m, n = costs.shape
    allocation = np.zeros((m, n))

    while np.sum(supply) > 0 and np.sum(demand) > 0:
        row_penalty = np.array([sorted(row[row != np.inf])[:2].ptp() if len(row[row != np.inf]) > 1 else 0 for row in costs])
        col_penalty = np.array([sorted(col[col != np.inf])[:2].ptp() if len(col[col != np.inf]) > 1 else 0 for col in costs.T])
        
        if row_penalty.max() >= col_penalty.max():
            i = row_penalty.argmax()
            j = np.argmin(costs[i])
        else:
            j = col_penalty.argmax()
            i = np.argmin(costs[:, j])

        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]

        if supply[i] == 0:
            costs[i, :] = np.inf
        if demand[j] == 0:
            costs[:, j] = np.inf

    return allocation


# Step 6: Transportation Simplex Algorithm (Placeholder)
def transportation_simplex(allocation, costs, supply, demand):
    m, n = allocation.shape  
    u = np.full(m, np.nan)  
    v = np.full(n, np.nan)  

    u[0] = 0  
    
    for i in range(m):
        for j in range(n):
            if allocation[i, j] > 0:
                if np.isnan(u[i]) and not np.isnan(v[j]):
                    u[i] = costs[i, j] - v[j]
                elif np.isnan(v[j]) and not np.isnan(u[i]):
                    v[j] = costs[i, j] - u[i]
                    
    while True:
        reduced_costs = costs - (u[:, np.newaxis] + v)
        if np.all(reduced_costs >= 0):
            print("Optimal solution found.")
            break
        i, j = np.unravel_index(np.argmin(reduced_costs), reduced_costs.shape)
        cycle = find_cycle(i, j, allocation) 
        delta = min([allocation[x, y] for x, y in cycle if allocation[x, y] > 0])
        
        for x, y in cycle:
            if allocation[x, y] > 0:
                allocation[x, y] -= delta
            else:
                allocation[x, y] += delta

        u[0] = 0  
        for i in range(m):
            for j in range(n):
                if allocation[i, j] > 0:
                    if np.isnan(u[i]) and not np.isnan(v[j]):
                        u[i] = costs[i, j] - v[j]
                    elif np.isnan(v[j]) and not np.isnan(u[i]):
                        v[j] = costs[i, j] - u[i]

    return allocation 

def find_cycle(i, j, allocation):
    m, n = allocation.shape
    visited = np.zeros_like(allocation, dtype=bool)
    cycle = []

    def dfs(x, y, path):
        if visited[x, y]:
            return False
        visited[x, y] = True
        path.append((x, y))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n:
                if (nx, ny) not in path:
                    if dfs(nx, ny, path):
                        return True
        path.pop()
        return False

    dfs(i, j, cycle)
    return cycle


# Main Function for Testing
if __name__ == "__main__":
    file_path = "C:/Users/DELL/Downloads/,Supplier1,Supplier2,Supplier3,Supp.txt"  
    data = load_parameters(file_path)

    costs = data.iloc[:-1, 1:].values  
    supply = data.iloc[:-1, 0].values  
    demand = data.iloc[-1, 1:].values  

    # Apply different methods to find the initial feasible solution
    nw_allocation = northwest_corner(supply.copy(), demand.copy())
    print("Northwest Corner Rule Allocation:\n", nw_allocation)

    mc_allocation = minimum_cost_method(costs.copy(), supply.copy(), demand.copy())
    print("Minimum Cost Method Allocation:\n", mc_allocation)

    mrc_allocation = minimum_row_cost_method(costs.copy(), supply.copy(), demand.copy())
    print("Minimum Row Cost Method Allocation:\n", mrc_allocation)

    v_allocation = vogels_method(costs.copy(), supply.copy(), demand.copy())
    print("Vogel's Method Allocation:\n", v_allocation)

    # Optimize using the Transportation Simplex Algorithm
    optimized_allocation = transportation_simplex(v_allocation, costs, supply, demand)
    print("Optimized Allocation using Transportation Simplex:\n", optimized_allocation)
