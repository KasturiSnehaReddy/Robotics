import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Warehouse Dimensions
MIN_X, MAX_X = -17, 17
MIN_Y, MAX_Y = -12, 12

def build_warehouse_graph():
    print("Building NetworkX Graph map of the warehouse...")
    G = nx.Graph()
    
    # 1. Create a grid of nodes
    for x in range(MIN_X + 1, MAX_X):
        for y in range(MIN_Y + 1, MAX_Y):
            G.add_node((x, y))
            
    # 2. Add edges (4-way connectivity)
    for x, y in G.nodes():
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx_pos in neighbors:
            if nx_pos in G:
                G.add_edge((x, y), nx_pos, weight=1.0)
                
    import os
    obstacles = []
    
    if os.path.exists("scanned_grid.npy"):
        print(">>> CLOSED LOOP: Loading 2D Occupancy Grid from LiDAR scan! <<<")
        occupancy_grid = np.load("scanned_grid.npy")
        for row in range(25):
            for col in range(35):
                if occupancy_grid[row, col] == 1:
                    x = col - 17
                    y = row - 12
                    obstacles.append((x, y))
    else:
        print(">>> EXPLORATION MODE: No saved map found. Using hardcoded blueprint. <<<")
        # Racks at y = -6, -2, 2, 6 and x = -10, -6, -2, 2, 6
        for y_center in [-6, -2, 2, 6]:
            for x_center in [-10, -6, -2, 2, 6]:
                for x in range(x_center-1, x_center+2):
                    obstacles.append((x, y_center))
        # Office block at 12, -7
        for x in range(11, 15):
            for y in range(-8, -5):
                obstacles.append((x, y))
        # Barrels at 12,13,14 and 7,8
        for x in [12, 13, 14]:
            for y in [7, 8]:
                obstacles.append((x, y))
            
    # CRITICAL: Always ensure Start and Goal nodes are never marked as obstacles 
    safe_nodes = [(-12, -8), (12, 4), (-12, 8), (12, -4)]
    obstacles = [obs for obs in obstacles if obs not in safe_nodes]
            
    # Remove obstacle nodes from the graph
    for obs in obstacles:
        if obs in G:
            G.remove_node(obs)
            
    return G, obstacles

def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def plot_warehouse_path(G, obstacles, path, start, goal):
    print("Plotting 2D Visualization with Matplotlib...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw Warehouse Walls
    warehouse_bounds = patches.Rectangle((MIN_X, MIN_Y), MAX_X - MIN_X, MAX_Y - MIN_Y, 
                                         linewidth=3, edgecolor='black', facecolor='none')
    ax.add_patch(warehouse_bounds)
    
    # Draw Shelves (Obstacles)
    for (ox, oy) in obstacles:
        # Draw a 1x1 block for each obstacle node
        rect = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='orange')
        ax.add_patch(rect)
        
    # Draw Loading Dock Area (Yellow)
    dock = patches.Rectangle((-4, -9.5), 8, 1, color='yellow', alpha=0.5)
    ax.add_patch(dock)

    # Plot the Graph Nodes (Walkable space)
    for node in G.nodes():
        ax.plot(node[0], node[1], marker='.', color='lightgray', markersize=2)

    # Plot Start and Goal
    ax.plot(start[0], start[1], marker='s', color='blue', markersize=12, label='Start')
    ax.plot(goal[0], goal[1], marker='*', color='red', markersize=16, label='Goal')

    # Plot the Calculated A* Path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, marker='o', color='green', linewidth=3, markersize=6, label='A* Path')

    ax.set_xlim(MIN_X - 1, MAX_X + 1)
    ax.set_ylim(MIN_Y - 1, MAX_Y + 1)
    ax.set_aspect('equal')
    ax.set_title("NetworkX A* Path Planning in Warehouse")
    ax.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def main():
    G, obstacles = build_warehouse_graph()
    
    # Define a Start point (near the loading dock) and a Goal point (deep in an aisle)
    start_node = (0, -8)
    goal_node = (3, 8)
    
    print(f"Calculating A* path from {start_node} to {goal_node}...")
    try:
        path = nx.astar_path(G, source=start_node, target=goal_node, heuristic=euclidean_distance)
        print(f"Path found! Length: {len(path)} steps.")
    except nx.NetworkXNoPath:
        print("ERROR: No path found!")
        path = []

    plot_warehouse_path(G, obstacles, path, start_node, goal_node)

if __name__ == "__main__":
    main()
