"""
This file contains the planning code for the grocery shopper.
All code written in this file was written by John Kim.
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.spatial import cKDTree

# Constants
MAP_WIDTH = 900
MAP_HEIGHT = 483
MAP_SCALE = 30  # pixels per meter
OBS_MARGIN = 8

# A* path planning method (unused, but left here for checklist purposes)
# def path_planner(map, start, end):
#      """
#      A* Pathfinding algorithm implementation.
 
#      Parameters:
#      - map (np.array): A 2D numpy array representing the map grid where 0 is free space and 1 is an obstacle.
#      - start (tuple): The starting coordinates as (x, y) on the map grid.
#      - end (tuple): The goal coordinates as (x, y) on the map grid.
 
#      Returns:
#      - list: A list of waypoints (tuples) from start to end representing the optimal path found, or an empty list if no path is found.
#      """
#      def heuristic(a, b):
#          """
#          Heuristic function for A* algorithm (Euclidean distance).
         
#          Parameters:
#          - a (tuple): A point (x1, y1).
#          - b (tuple): A point (x2, y2).
         
#          Returns:
#          - float: The Euclidean distance between points a and b.
#          """
#          return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
     
#      movements = [
#          (0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1),  # Cardinal directions
#          (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)  # Diagonal directions
#      ]
     
#      open_set = {start}  # Set of nodes to be evaluated
#      closed_set = set()  # Set of nodes already evaluated
#      came_from = {}  # Tracks the best path to each node
#      g_score = {start: 0}  # Cost from start to current node
#      f_score = {start: heuristic(start, end)}  # Estimated cost from start to end
 
#      while open_set:
#          current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
#          if current == end:
#              # Reconstruct the path by tracing back from the end node
#              path = []
#              while current in came_from:
#                  path.append(current)
#                  current = came_from[current]
#              path.append(start)
#              path.reverse()  # Path is reversed to go from start to end
#              return path
#          open_set.remove(current)
#          closed_set.add(current)
         
#          for dx, dy, cost in movements:
#              neighbor = (current[0] + dx, current[1] + dy)
#              # Check if the neighbor is within bounds and not an obstacle
#              if (0 <= neighbor[0] < map.shape[1] and
#                  0 <= neighbor[1] < map.shape[0] and
#                  neighbor not in closed_set and
#                  map[neighbor[1]][neighbor[0]] == 0):
#                  tentative_g = g_score[current] + cost
#                  if neighbor not in open_set:
#                      open_set.add(neighbor)
#                  elif tentative_g >= g_score.get(neighbor, float('inf')):
#                      continue
#                  came_from[neighbor] = current
#                  g_score[neighbor] = tentative_g
#                  f_score[neighbor] = tentative_g + heuristic(neighbor, end)
     
#      print("No path found")
#      return []

# Function to inflate obstacles in the map
def inflate_map(map_array):
    """
    Inflates obstacles in the map by a given radius to ensure safe paths for larger robots.
    """
    # Create a circular kernel for inflation
             
    robot_radius = int(.40 * 30)  # 0.35m radius, scaled by 30 px/m
    kernel = np.zeros((2*robot_radius+1, 2*robot_radius+1))
    y, x = np.ogrid[-robot_radius:robot_radius+1, -robot_radius:robot_radius+1]
    mask = x**2 + y**2 <= robot_radius**2
    kernel[mask] = 1
    
    cspace = convolve2d(map_array, kernel, mode='same') > 0
    cspace = cspace.astype(np.uint8)  # Convert to binary map

    return cspace
    
# Function to visualize path on map
def visualize_path(path, map_array_rgb, step=1, radius=2):
    """
    Paints a red dot (square brush) around each waypoint.
    'radius' controls how big the dot is.
    """
    print(f"Visualizing path with {len(path)} waypoints.")
    for i in range(0, len(path), step):
        map_x, map_y = path[i]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                px = map_x + dx
                py = map_y + dy
                if 0 <= px < map_array_rgb.shape[1] and 0 <= py < map_array_rgb.shape[0]:
                    map_array_rgb[py, px] = [1.0, 0.0, 0.0]  # RED
    return map_array_rgb

def world_to_map_coords(world_x, world_y, scale_x=30, scale_y=30, map_width=900, map_height=483):
    # Offset origin to center of the map
    map_x = int((world_x * scale_x) + map_width // 2)
    map_y = int(map_height // 2 - (world_y * scale_y))  # Y flipped

    # Debug the conversion results
    # print(f"World ({world_x}, {world_y}) → Map ({map_x}, {map_y})") # Keep commented unless debugging

    # Ensure coordinates are within map bounds
    map_x = min(max(0, map_x), map_width - 1)
    map_y = min(max(0, map_y), map_height - 1)

    return map_x, map_y

def map_to_world_coords(map_x, map_y, scale_x=30, scale_y=30, map_width=900, map_height=483):
    """Converts map pixel coordinates back to world coordinates (meters)."""
    world_x = (map_x - map_width // 2) / scale_x
    world_y = (map_height // 2 - map_y) / scale_y  # Y flipped back

    # print(f"Map ({map_x}, {map_y}) → World ({world_x:.2f}, {world_y:.2f})") # Keep commented unless debugging
    return world_x, world_y

# Bresenham's line algorithm to check if the path between two points is free of obstacles
def bresenham(x0, y0, x1, y1):
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

# Check if the line between two points is free (i.e., no obstacles)
def is_free(p1, p2, map_array, margin=0, threshold=0.3):
    """
    Check if the path between two points is free from obstacles.
    Margin can be smaller now because we're using pre-inflated obstacles.
    """
    for x, y in bresenham(*p1, *p2):
        # We can use a smaller margin here since the map is pre-inflated
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                px = x + dx
                py = y + dy
                if 0 <= px < map_array.shape[1] and 0 <= py < map_array.shape[0]:
                    if map_array[py, px] > threshold:
                        return False
    return True

# Calculate clearance for a point
def calculate_clearance(point, map_array, max_radius=20, threshold=0.3):
    """
    Calculate how far the nearest obstacle is from this point.
    Returns a value between 0 (obstacle) and max_radius (clear space).
    """
    x, y = point
    
    # Quick check if point is on an obstacle
    if map_array[y, x] > threshold:
        return 0
        
    # Check in expanding circles
    for r in range(1, max_radius + 1):
        found_obstacle = False
        
        # Check only the perimeter of the circle (approximately)
        for angle in np.linspace(0, 2*np.pi, 8*r):  # More samples for larger circles
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            
            if 0 <= px < map_array.shape[1] and 0 <= py < map_array.shape[0]:
                if map_array[py, px] > threshold:
                    return r
    
    return max_radius

# Ensure full path clearance (not just endpoints)
def path_clearance_ok(p1, p2, map_array, min_clearance):
    """
    Ensures the entire edge from p1 to p2 maintains a safe clearance from obstacles.
    """
    for x, y in bresenham(*p1, *p2):
        if calculate_clearance((x, y), map_array) < min_clearance:
            return False
    return True

# RRT Pathfinding Implementation
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def steer(from_node, to_node, step_size):
    direction = np.array(to_node) - np.array(from_node)
    length = np.linalg.norm(direction)
    if length == 0:
        return from_node
    direction = direction / length
    new_point = np.array(from_node) + step_size * direction
    return tuple(np.round(new_point).astype(int))

def sample_point(map_array, goal, iteration, max_iter, goal_sample_rate=0.2, clearance_bias=0.3):
    """
    Advanced sampling strategy:
    - Dynamic goal sampling rate
    - Bias toward areas with higher clearance
    """
    # Increase goal bias as iterations progress
    dynamic_goal_rate = goal_sample_rate + (0.3 * iteration / max_iter)
    
    if random.random() < dynamic_goal_rate:
        return goal
    
    # Sample random point
    for _ in range(10):  # Try a few times to find a point with good clearance
        x = random.randint(0, map_array.shape[1] - 1)
        y = random.randint(0, map_array.shape[0] - 1)
        
        # If point is not on an obstacle, consider it
        if map_array[y, x] < 0.3:
            if random.random() > clearance_bias:
                return (x, y)
            
            # With some probability, check clearance and accept if it's good
            clearance = calculate_clearance((x, y), map_array)
            if clearance > 5:  # Prefer points with good clearance
                return (x, y)
    
    # Fallback to basic random sampling
    return (
        random.randint(0, map_array.shape[1] - 1),
        random.randint(0, map_array.shape[0] - 1),
    )

def adaptive_step_size(from_node, current_step, map_array, min_step=5, max_step=15):
    """
    Adjust step size based on environment complexity.
    Smaller steps in cluttered areas, larger steps in open space.
    """
    clearance = calculate_clearance(from_node, map_array)
    
    # Map clearance to step size, with bounds
    step = min_step + (clearance / 10) * (max_step - min_step)
    return min(max(step, min_step), max_step)

# Basic RRT Implementation
def rrt_basic(map_array, start, goal, max_iter=3000, step_size=15, goal_reach_threshold=20):
    """
    Basic RRT Algorithm Implementation.
    """
    nodes = [start]
    parents = {start: None}
    tree = cKDTree([start]) # Use KDTree for efficient nearest neighbor search

    print(f"Starting Basic RRT with start: {start} and goal: {goal}")

    # Verify start and goal are valid
    if map_array[start[1], start[0]] > 0.3:
        print("ERROR: Start point is inside an obstacle!")
        return []
    if map_array[goal[1], goal[0]] > 0.3:
        print("ERROR: Goal point is inside an obstacle!")
        return []

    for i in range(max_iter):
        # Sample a random point (with occasional goal bias)
        if random.random() < 0.1: # 10% chance to sample goal
            rand_point = goal
        else:
            rand_point = (
                random.randint(0, map_array.shape[1] - 1),
                random.randint(0, map_array.shape[0] - 1),
            )

        # Find nearest node
        _, nearest_idx = tree.query([rand_point], k=1)
        if isinstance(nearest_idx, (np.ndarray, list)):
             nearest_idx = nearest_idx[0] # Ensure scalar index
        nearest_node = nodes[nearest_idx]

        # Steer towards the random point
        new_node = steer(nearest_node, rand_point, step_size)

        # Check if new_node is valid (within bounds and not in collision)
        if not (0 <= new_node[0] < map_array.shape[1] and 0 <= new_node[1] < map_array.shape[0]) \
           or map_array[new_node[1], new_node[0]] > 0.3:
            continue

        # Check if the path from nearest_node to new_node is collision-free
        if is_free(nearest_node, new_node, map_array, margin=0):
            nodes.append(new_node)
            parents[new_node] = nearest_node
            tree = cKDTree(nodes) # Rebuild tree after adding node

            # Check if goal is reachable from the new_node
            dist_to_goal = distance(new_node, goal)
            if dist_to_goal < goal_reach_threshold and is_free(new_node, goal, map_array, margin=0):
                parents[goal] = new_node
                print(f"Goal reached at iteration {i}!")

                # Reconstruct path
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = parents.get(node) # Use .get for safety
                path.reverse()
                print(f"Basic RRT path found with {len(path)} waypoints.")
                return path

        if i % 500 == 0: # Print status periodically
             print(f"Basic RRT Iteration {i}/{max_iter}, Nodes: {len(nodes)}")


    print("WARNING: Basic RRT reached max iterations without finding a path to the goal.")
    return [] # No path found within max_iter

# Keep RRT* implementation below (or comment out/remove if preferred)
def rrt_star(map_array, start, goal, max_iter=3000, base_step_size=10, goal_sample_rate=0.2, search_radius_factor=1.5, rebuild_tree_freq=100): # Added rebuild_tree_freq
    """
    RRT* Algorithm:
    - Advanced sampling
    - Adaptive step size
    - Cost calculation and propagation
    - Rewiring the tree for optimal paths
    """
    nodes = [start]
    parents = {start: None}
    costs = {start: 0.0}  # Cost from start to node

    print(f"Starting RRT* with start: {start} and goal: {goal}")

    # Verify start and goal are valid
    if map_array[start[1], start[0]] == 1:
        print("ERROR: Start point is inside an obstacle!")
        return []

    if map_array[goal[1], goal[0]] == 1:
        print("ERROR: Goal point is inside an obstacle!")
        return []

    tree = cKDTree([start])

    for i in range(max_iter):
        # Sample a random point with advanced strategy
        rand_point = sample_point(map_array, goal, i, max_iter, goal_sample_rate)

        # Find nearest node using the current tree
        _, nearest_idx = tree.query([rand_point], k=1)
        # Ensure index is scalar if k=1
        if isinstance(nearest_idx, (np.ndarray, list)):
             nearest_idx = nearest_idx[0]
        nearest_node = nodes[nearest_idx]

        # Use adaptive step size
        step_size = adaptive_step_size(nearest_node, base_step_size, map_array)

        # Steer towards the random point
        new_node = steer(nearest_node, rand_point, step_size)

        # Skip if new_node is out of bounds or in collision
        if not (0 <= new_node[0] < map_array.shape[1] and 0 <= new_node[1] < map_array.shape[0]) \
           or map_array[new_node[1], new_node[0]] > 0.3:
            continue

        # --- RRT* Specific Logic ---

        # Calculate search radius (dynamic based on number of nodes)
        num_nodes = len(nodes)
        dimension = 2
        # Heuristic: gamma * (log(n)/n)^(1/d)
        # Adjusted gamma for practical performance
        gamma = base_step_size * 10 # Tuning parameter
        search_radius = min(gamma * (math.log(num_nodes + 1) / (num_nodes + 1))**(1./dimension) * search_radius_factor, step_size * 3) # Cap radius
        search_radius = max(search_radius, step_size) # Minimum radius

        # Find near nodes within the search radius using the current tree
        near_indices = tree.query_ball_point(new_node, search_radius)
        near_nodes = [nodes[k] for k in near_indices]

        # Choose Parent: Select the best parent from near_nodes
        best_parent = nearest_node
        min_cost = costs[nearest_node] + distance(nearest_node, new_node)

        # Check connection to potential parents first
        possible_parents = [(costs[near_node] + distance(near_node, new_node), near_node) for near_node in near_nodes if is_free(near_node, new_node, map_array, margin=0)]
        # Add the connection via the nearest node if it's free
        if is_free(nearest_node, new_node, map_array, margin=0):
             possible_parents.append((min_cost, nearest_node))

        # If no valid parent connection found, continue
        if not possible_parents:
             continue

        # Find the best parent among valid connections
        min_cost, best_parent = min(possible_parents, key=lambda x: x[0])

        # Add the new node
        nodes.append(new_node)
        parents[new_node] = best_parent
        costs[new_node] = min_cost

        # Rewire: Check if new_node offers a better path for near nodes
        for k in near_indices: # Iterate using indices
             near_node = nodes[k]
             if near_node == best_parent: continue # Don't rewire the parent itself

             cost_via_new = costs[new_node] + distance(new_node, near_node)
             if cost_via_new < costs[near_node] and is_free(new_node, near_node, map_array, margin=0):
                parents[near_node] = new_node
                costs[near_node] = cost_via_new
                # Note: Cost updates should propagate in full RRT*, but omitted here.

        # Check if goal is reachable from the new_node
        dist_to_goal = distance(new_node, goal)
        if dist_to_goal < step_size * 1.5 and is_free(new_node, goal, map_array, margin=0):
            cost_to_goal_via_new = costs[new_node] + dist_to_goal
            # Update goal connection only if this path is better than any existing path to goal
            if cost_to_goal_via_new < costs.get(goal, float('inf')):
                parents[goal] = new_node
                costs[goal] = cost_to_goal_via_new
                print(f"Goal potentially reached at iteration {i} with cost {cost_to_goal_via_new:.2f}")
                # RRT* continues searching

        # --- Rebuild KDTree periodically ---
        if i % rebuild_tree_freq == 0:
            tree = cKDTree(nodes) # Rebuild the tree

        if i % 100 == 0:  # Print status every 100 iterations
            # Update tree before printing status if it wasn't just rebuilt
            if i % rebuild_tree_freq != 0:
                 tree = cKDTree(nodes)
            closest_dist = min([distance(n, goal) for n in nodes]) if nodes else float('inf')
            print(f"Iteration {i}/{max_iter}, Nodes: {len(nodes)}, Closest distance to goal: {closest_dist:.2f}, Goal cost: {costs.get(goal, float('inf')):.2f}")

    # --- Path Reconstruction ---
    # Ensure the tree is up-to-date before final checks
    tree = cKDTree(nodes)

    # If we reached max iterations without finding the goal node in parents
    if goal not in parents:
        print("WARNING: RRT* reached max iterations without directly connecting to goal.")
        # Find the node closest to the goal that we *can* reach
        reachable_nodes = {n: costs[n] for n in nodes if n in parents}
        if not reachable_nodes:
             print("ERROR: No reachable nodes found!")
             return []
        closest_node = min(reachable_nodes, key=lambda n: distance(n, goal))
        
        # Check if a final connection is possible
        if is_free(closest_node, goal, map_array, margin=0):
            parents[goal] = closest_node
            costs[goal] = costs[closest_node] + distance(closest_node, goal)
            print(f"Connected to goal via closest node {closest_node} at cost {costs[goal]:.2f}")
        else:
            print("ERROR: Could not find valid path to goal even after checking closest node.")
            return [] # No valid path found
    
    # Reconstruct the path from the goal using the parent pointers
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()

    print(f"Optimal path found with {len(path)} waypoints and cost {costs[goal]:.2f}")
    return path

def smooth_path(path, map_array, smoothing_radius=2):
    """
    Simplify the path by removing redundant waypoints while ensuring 
    the path remains obstacle-free.
    """
    if len(path) <= 2:
        return path
    
    smoothed = [path[0]]
    i = 0
    
    while i < len(path) - 1:
        max_jump = len(path) - 1   # try to connect straight to the end
        connected = False
        for j in range(max_jump, i, -1):
            if is_free(path[i], path[j], map_array, margin=smoothing_radius) and path_clearance_ok(path[i], path[j], map_array, OBS_MARGIN):
                smoothed.append(path[j])
                i = j
                connected = True
                break
        if not connected:
            i += 1
            if i < len(path):
                smoothed.append(path[i])
    
    print(f"Smoothed path from {len(path)} to {len(smoothed)} waypoints")
    return smoothed

def convert_paths_to_waypoints(paths):
    """
    Convert a list of paths (segments) into a list of waypoints.
    """
    waypoints = []
    for path in paths:
        for p in path:
            waypoints.append(p)
    return waypoints

def visualize_map_and_path_on_display(display, map_array, path, path_color=0xFF0000, obstacle_color=0xFFFFFF, free_space_color=0x000000, obstacle_threshold=0.3):
    """
    Visualizes the map and the path on the robot's display device.

    Args:
        display: The robot's display device.
        map_array: The 2D numpy array representing the map (values > threshold are obstacles).
        path: A list of (x, y) tuples representing the path in map coordinates.
        path_color: The color to draw the path pixels (default: red).
        obstacle_color: The color for obstacles (default: white).
        free_space_color: The color for free space (default: black).
        obstacle_threshold: The value above which a map cell is considered an obstacle.
    """
    if display is None:
        print("Error: Display device not provided for visualization.")
        return

    height, width = map_array.shape
    print(f"Visualizing map ({width}x{height}) and path ({len(path)} points) on display...")

    # Draw the map background
    for y in range(height):
        for x in range(width):
            if map_array[y, x] > obstacle_threshold:
                display.setColor(int(obstacle_color))
            else:
                display.setColor(int(free_space_color))
            display.drawPixel(x, y)

    # Draw the path
    if path:
        display.setColor(int(path_color))
        for point in path:
            x, y = point
            # Ensure coordinates are within bounds before drawing
            if 0 <= x < width and 0 <= y < height:
                display.drawPixel(x, y)
            # else:
            #     print(f"Warning: Path point ({x}, {y}) out of bounds ({width}x{height}) for display.") # Optional warning

    print("Display visualization complete.")

# Random shortcut to remove zigzags from the path
def random_shortcut(path, map_array, tries=2000):
    import random
    new_path = path[:]
    for _ in range(tries):
        if len(new_path) < 3: break
        i, j = sorted(random.sample(range(len(new_path)), 2))
        if j - i < 2: continue
        if is_free(new_path[i], new_path[j], map_array, margin=OBS_MARGIN):
            # splice out the middle
            new_path = new_path[:i+1] + new_path[j:]
    return new_path

def prune_collinear(path, angle_tol=5.0):
    pruned = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:]):
        v1 = np.array(b) - np.array(a)
        v2 = np.array(c) - np.array(b)
        ang = np.degrees(np.arccos(np.dot(v1, v2) /
                       (np.linalg.norm(v1)*np.linalg.norm(v2))))
        if abs(ang - 180) > angle_tol:
            pruned.append(tuple(b))
    pruned.append(path[-1])
    return pruned

# Main Planner Code
def main():
    # Start position based on actual robot's initial position
    start_w = (0, 0)  # Hard-coded world coordinates of robot's position

    # Define multiple waypoints to visit in sequence
    waypointList = [(0, 0), (6.33057, 8.99417), (0.948708, 10.4302), (0, 0)]  # Example waypoints

    # Load map
    map_array = np.load("map.npy")
    
    # Pre-inflate the obstacles in the map
    inflated_map = inflate_map(map_array)

    #quickly display the inflated map
    plt.imshow(inflated_map)
    plt.show()

    # Create a list to store all path segments
    all_paths = []

    # Process each segment of the path
    for i in range(len(waypointList) - 1):
        current_start = waypointList[i]
        current_end = waypointList[i+1]

        # Convert the start and end from the webots coordinate frame into the map frame
        start = world_to_map_coords(*current_start)
        end = world_to_map_coords(*current_end)

        # Plan path for this segment using RRT*
        path = rrt_star(inflated_map, start, end, max_iter=5000)
        
        if path:
            # 1) greedy prune
            smoothed = smooth_path(path, inflated_map)
            # 2) random shortcut to knock out all the little arcs
            smoothed = random_shortcut(smoothed, inflated_map, tries=500)
            # 3) prune near-collinear points
            smoothed = prune_collinear(smoothed)
            print(f"Segment {i+1} → {len(path)}→{len(smoothed)} after greedy+srand+prune shortcut")
            all_paths.append(smoothed)
        else:
            print(f"Failed to find path for segment {i+1} — skipping.")
            break

    # Convert paths to waypoints and save on disk as path.npy
    if all_paths:
        all_waypoints = convert_paths_to_waypoints(all_paths)
        if all_waypoints:
            np.save("path.npy", np.array(all_waypoints))
            print(f"Complete path saved with {len(all_waypoints)} waypoints to path.npy")
        
        else:
            print("ERROR: No waypoints were generated! Path following will not work.")

if __name__ == "__main__":
    main()
