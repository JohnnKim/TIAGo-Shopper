import numpy as np
import math

def follow_waypoints(path, robot_parts, vL, vR, waypoint_index, pose_theta, pose_x=None, pose_y=None, gps=None):
    """
    Makes the robot follow a path of waypoints using an enhanced controller.
    
    Args:
        path: List of waypoints in pixel coordinates [(x1, y1), (x2, y2), ...]
        robot_parts: Dictionary of robot motor parts
        vL, vR: Current left and right wheel velocities
        waypoint_index: Current waypoint index the robot is targeting
        pose_theta: Current orientation of the robot (radians)
        pose_x, pose_y: Current position of the robot in world coordinates (optional)
        gps: GPS device (optional)
        
    Returns:
        vL, vR: Updated wheel velocities
        waypoint_index: Updated waypoint index
    """
    # Constants for the controller
    MAX_SPEED = 7.0  # [rad/s]
    MAP_WIDTH = 900
    MAP_HEIGHT = 483
    MAP_SCALE = 30  # pixels per meter
    
    # Get current position if not provided
    if pose_x is None or pose_y is None:
        if gps is not None:
            gps_values = gps.getValues()
            pose_x = gps_values[0]
            pose_y = gps_values[1]
        else:
            # If neither position nor GPS is provided, we can't navigate
            print("Warning: No position information available for path following")
            return vL, vR, waypoint_index
    
    # Skip if path is empty
    if not path or len(path) == 0:
        print("Warning: Path is empty or None")
        return 0, 0, waypoint_index
    
    # Check if we've reached the end of the path
    if waypoint_index >= len(path):
        print("Path completed! Stopping robot.")
        return 0, 0, waypoint_index  # Stop the robot
    
    # Get current target waypoint (in pixel coordinates)
    target_pixel_x, target_pixel_y = path[waypoint_index]
    
    # Convert target from pixel coordinates to world coordinates
    target_world_x = (target_pixel_x - MAP_WIDTH/2) / MAP_SCALE
    target_world_y = (MAP_HEIGHT/2 - target_pixel_y) / MAP_SCALE
    
    # Calculate distance to target
    dx = target_world_x - pose_x
    dy = target_world_y - pose_y
    distance = math.sqrt(dx**2 + dy**2)
    
    # Calculate angle to target
    target_angle = math.atan2(dy, dx)
    
    # Calculate heading error (shortest angle)
    angle_error = math.atan2(math.sin(target_angle - pose_theta), 
                             math.cos(target_angle - pose_theta))
    
    # Logging (less frequent to reduce console spam)
    if waypoint_index % 10 == 0 or distance < 0.5:
        print(f"Following waypoint {waypoint_index}/{len(path)-1}")
        print(f"Target: world ({target_world_x:.2f}, {target_world_y:.2f})")
        print(f"Current: ({pose_x:.2f}, {pose_y:.2f}), heading: {math.degrees(pose_theta):.1f}°")
        print(f"Distance: {distance:.2f}m, Angle error: {math.degrees(angle_error):.1f}°")
    
    # Dynamic waypoint threshold - smaller for final waypoint
    is_final_waypoint = waypoint_index == len(path) - 1
    waypoint_threshold = 0.15 if is_final_waypoint else 0.3  # meters
    
    # Look-ahead mechanism for smoother following
    look_ahead = min(3, len(path) - waypoint_index - 1)
    if look_ahead > 0 and distance < 0.8 and not is_final_waypoint:
        # Look at a future waypoint for smoother trajectory
        future_pixel_x, future_pixel_y = path[waypoint_index + look_ahead]
        future_world_x = (future_pixel_x - MAP_WIDTH/2) / MAP_SCALE
        future_world_y = (MAP_HEIGHT/2 - future_pixel_y) / MAP_SCALE
        
        # Blend current and future targets based on distance
        blend_factor = min(1.0, distance * 2)  # 0 when close, 1 when far
        target_world_x = blend_factor * target_world_x + (1 - blend_factor) * future_world_x
        target_world_y = blend_factor * target_world_y + (1 - blend_factor) * future_world_y
        
        # Recalculate with blended target
        dx = target_world_x - pose_x
        dy = target_world_y - pose_y
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        angle_error = math.atan2(math.sin(target_angle - pose_theta), 
                                math.cos(target_angle - pose_theta))
    
    # Check if we've reached the current waypoint
    if distance < waypoint_threshold:
        waypoint_index += 1
        print(f"Reached waypoint {waypoint_index-1}, moving to next one")
        
        # If we've reached the final waypoint, stop the robot
        if waypoint_index >= len(path):
            print("Final waypoint reached! Path completed.")
            return 0, 0, waypoint_index
            
        return vL, vR, waypoint_index
    
    # Enhanced proportional controller with dynamic gains
    Kp_distance = 3.0
    Kp_angle = 2.0
    
    # Adjust gains based on distance and angle error
    if abs(angle_error) > math.radians(30):
        # Prioritize turning when angle error is large
        Kp_distance *= 0.5
        Kp_angle *= 1.5

    # Now limit the angular correction even more
    angular_correction = Kp_angle * angle_error

    # Cap angular velocity for smoother turns (previous max was 0.6, now we reduce that)
    max_angular = MAX_SPEED * 0.4  # Reduced angular velocity cap
    angular_correction = max(min(angular_correction, max_angular), -max_angular)

    # Base speed depends on distance and angle error
    base_speed = min(MAX_SPEED * 0.7, Kp_distance * distance)

    # Ensure minimum speed but slow down for final approach
    if is_final_waypoint and distance < 1.0:
        base_speed = max(base_speed * 0.7, 1.0)  # Slower final approach
    else:
        base_speed = max(base_speed, 2.0)  # Minimum speed otherwise

    # Compute wheel velocities
    vL = base_speed - angular_correction
    vR = base_speed + angular_correction
    
    # Limit acceleration for smoother movement
    max_speed_change = 1.0  # Maximum change in speed per update
    vL = min(max(vL, vL - max_speed_change), vL + max_speed_change)
    vR = min(max(vR, vR - max_speed_change), vR + max_speed_change)
    
    # Ensure velocities are within limits
    vL = min(max(vL, -MAX_SPEED), MAX_SPEED)
    vR = min(max(vR, -MAX_SPEED), MAX_SPEED)
    
    return vL, vR, waypoint_index