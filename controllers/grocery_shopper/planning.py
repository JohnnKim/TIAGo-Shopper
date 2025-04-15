import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.signal import convolve2d

# A* Implementation
def path_planner(map, start, end):
    """
    A* Pathfinding algorithm implementation.

    Parameters:
    - map (np.array): A 2D numpy array representing the map grid where 0 is free space and 1 is an obstacle.
    - start (tuple): The starting coordinates as (x, y) on the map grid.
    - end (tuple): The goal coordinates as (x, y) on the map grid.

    Returns:
    - list: A list of waypoints (tuples) from start to end representing the optimal path found, or an empty list if no path is found.
    """
    def heuristic(a, b):
        """
        Heuristic function for A* algorithm (Euclidean distance).
        
        Parameters:
        - a (tuple): A point (x1, y1).
        - b (tuple): A point (x2, y2).
        
        Returns:
        - float: The Euclidean distance between points a and b.
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    movements = [
        (0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1),  # Cardinal directions
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)  # Diagonal directions
    ]
    
    open_set = {start}  # Set of nodes to be evaluated
    closed_set = set()  # Set of nodes already evaluated
    came_from = {}  # Tracks the best path to each node
    g_score = {start: 0}  # Cost from start to current node
    f_score = {start: heuristic(start, end)}  # Estimated cost from start to end

    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        if current == end:
            # Reconstruct the path by tracing back from the end node
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()  # Path is reversed to go from start to end
            return path
        open_set.remove(current)
        closed_set.add(current)
        
        for dx, dy, cost in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            # Check if the neighbor is within bounds and not an obstacle
            if (0 <= neighbor[0] < map.shape[1] and
                0 <= neighbor[1] < map.shape[0] and
                neighbor not in closed_set and
                map[neighbor[1]][neighbor[0]] == 0):
                tentative_g = g_score[current] + cost
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
    
    print("No path found")
    return []

# RRT Implementation
def distance(a, b):
    """
    Calculates the Euclidean distance between two points.
    
    Parameters:
    - a (tuple): First point (x1, y1).
    - b (tuple): Second point (x2, y2).
    
    Returns:
    - float: The Euclidean distance between a and b.
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def steer(from_node, to_node, step_size):
    """
    Steers the robot from one node to another using a fixed step size.
    
    Parameters:
    - from_node (tuple): The current node (x1, y1).
    - to_node (tuple): The target node (x2, y2).
    - step_size (int): The step size used for movement.
    
    Returns:
    - tuple: The new node (x, y) after steering from the current node.
    """
    direction = np.array(to_node) - np.array(from_node)
    length = np.linalg.norm(direction)
    if length == 0:
        return from_node
    direction = direction / length
    new_point = np.array(from_node) + step_size * direction
    return tuple(np.round(new_point).astype(int))

def bresenham(x0, y0, x1, y1):
    """
    Bresenham's line algorithm for generating a list of points between two coordinates.
    
    Parameters:
    - x0 (int): The starting x-coordinate.
    - y0 (int): The starting y-coordinate.
    - x1 (int): The ending x-coordinate.
    - y1 (int): The ending y-coordinate.
    
    Returns:
    - list: A list of points along the line between (x0, y0) and (x1, y1).
    """
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def is_free(p1, p2, map_array):
    """
    Checks if the line between two points is free of obstacles.
    
    Parameters:
    - p1 (tuple): First point (x1, y1).
    - p2 (tuple): Second point (x2, y2).
    - map_array (np.array): The map array with 0 for free space and 1 for obstacles.
    
    Returns:
    - bool: True if the path between the points is free, False otherwise.
    """
    for x, y in bresenham(*p1, *p2):
        if not (0 <= x < map_array.shape[1] and 0 <= y < map_array.shape[0]):
            return False
        if map_array[y][x] > 0:
            return False
    return True

def rrt(map_array, start, goal, max_iter=1000, step_size=10, goal_sample_rate=0.1):
    """
    Rapidly-exploring Random Tree (RRT) pathfinding algorithm.
    
    Parameters:
    - map_array (np.array): The 2D map array with 0 for free space and 1 for obstacles.
    - start (tuple): The starting coordinates as (x, y)
    - goal (tuple): The goal coordinates as (x, y)
    - max_iter (int): The maximum number of iterations to try for pathfinding.
    - step_size (int): The step size for exploring the space.
    - goal_sample_rate (float): The probability of sampling the goal directly instead of random points.
    
    Returns:
    - list: A list of waypoints (tuples) from start to goal if found, otherwise an empty list.
    """
    nodes = [start]
    parents = {start: None}
    
    for _ in range(max_iter):
        if random.random() < goal_sample_rate:
            rand_point = goal
        else:
            rand_point = (
                random.randint(0, map_array.shape[1] - 1),
                random.randint(0, map_array.shape[0] - 1),
            )
        nearest = min(nodes, key=lambda n: distance(n, rand_point))
        new_node = steer(nearest, rand_point, step_size)
        
        if is_free(nearest, new_node, map_array):
            nodes.append(new_node)
            parents[new_node] = nearest
            if distance(new_node, goal) < step_size and is_free(new_node, goal, map_array):
                parents[goal] = new_node
                break
    
    path = []
    node = goal if goal in parents else min(nodes, key=lambda n: distance(n, goal))
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    return path

def smooth_path(path, map_array):
    """
    Smooths the path by removing unnecessary waypoints.
    
    Parameters:
    - path (list): The list of waypoints to be smoothed.
    - map_array (np.array): The map array to check if a segment is free of obstacles.
    
    Returns:
    - list: A list of smoothed waypoints.
    """
    if not path:
        return []
    smooth = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i:
            if is_free(path[i], path[j], map_array):
                smooth.append(path[j])
                i = j
                break
            j -= 1
    return smooth

# Waypoint Conversion
def convert_paths_to_waypoints(all_paths, density_level=0):
    """
    Converts the path into waypoints with adjustable density.
    
    Parameters:
    - all_paths (list): A list of paths (each path is a list of waypoints).
    - density_level (int): Determines the density of the waypoints.
    
    Returns:
    - list: A list of waypoints representing the converted paths.
    """
    all_waypoints = []
    for segment_idx, path in enumerate(all_paths):
        if density_level == 0:
            first_spacing, middle_spacing, last_spacing = 30, 20, 10
        else:
            first_spacing, middle_spacing, last_spacing = 15, 10, 5

        first_portion = min(len(path) // 3, 100)
        second_portion = min(2 * len(path) // 3, 200)
        segment_waypoints = []

        for i in range(0, first_portion, first_spacing):
            point = path[i]
            segment_waypoints.append(_map_to_world(point))
        for i in range(first_portion, second_portion, middle_spacing):
            point = path[i]
            segment_waypoints.append(_map_to_world(point))
        for i in range(second_portion, len(path), last_spacing):
            point = path[i]
            segment_waypoints.append(_map_to_world(point))

        # Always include end
        last_point = path[-1]
        end_wp = _map_to_world(last_point)
        if end_wp not in segment_waypoints:
            segment_waypoints.append(end_wp)

        if segment_idx > 0 and segment_waypoints:
            all_waypoints.extend(segment_waypoints[1:])
        else:
            all_waypoints.extend(segment_waypoints)

    if all_waypoints:
        all_waypoints.pop(0)
    return all_waypoints

def _map_to_world(point):
    """
    Converts map coordinates to world coordinates.
    
    Parameters:
    - point (tuple): A map coordinate (x, y).
    
    Returns:
    - tuple: The corresponding world coordinates (wx, wy).
    """
    x, y = point
    wx = -((360 - x) / 30.0)
    wy = -((y) / 30.0)
    return (wx, wy)

