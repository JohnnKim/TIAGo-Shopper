import numpy as np
import math

"""
Entry point for odometry
"""
def main(robot, compass, AXLE_LENGTH, gps, display, vL, vR, prev_time, pose_x, pose_y, pose_theta):
    # Validate parameters
    if abs(AXLE_LENGTH) < 1e-6:
        print("Warning: AXLE_LENGTH is too small, using default value")
        AXLE_LENGTH = 0.1  # Default safe value
    
    # Process timing
    current_time = robot.getTime()
    dt = current_time - prev_time
    if dt < 1e-6:  # Avoid division by very small numbers
        return pose_x, pose_y, pose_theta, current_time
    
    # Get sensor readings
    gps_position = gps.getValues()
    compass_heading = get_heading(compass)
    
    # Update odometry
    pose_x, pose_y, pose_theta = update_odometry(vL, vR, dt, pose_x, pose_y, pose_theta, AXLE_LENGTH)
    
    # Normalize pose_theta to prevent unbounded growth
    pose_theta = normalize_angle(pose_theta)
    
    # Simple sensor fusion - occasional reset from absolute sensors to prevent drift
    # Only apply corrections when moving (to avoid GPS noise when stationary)
    if abs(vL) > 0.01 or abs(vR) > 0.01:
        # Heading correction from compass (subtle to avoid jumps)
        heading_error = normalize_angle(compass_heading - pose_theta)
        pose_theta += 0.05 * heading_error  # Small correction factor
        
        # Optional position correction from GPS (use sparingly to maintain odometry smoothness)
        if robot.getTime() % 10 < 0.1:  # Apply correction roughly every 10 seconds
            pose_x = 0.9 * pose_x + 0.1 * gps_position[0]  # 10% GPS, 90% odometry
            pose_y = 0.9 * pose_y + 0.1 * gps_position[2]  # Using z from GPS as y in 2D
    
    # Update map with current position (converting from world to map coordinates)
    #update_map(map, pose_x, pose_y)
    
    # Display data for debugging
    print(f"GPS: ({gps_position[0]:.2f}, {gps_position[2]:.2f}), Heading: {compass_heading:.2f}, "
          f"Odometry: (x: {pose_x:.2f}, y: {pose_y:.2f}, θ: {pose_theta:.2f})")
    
    return pose_x, pose_y, pose_theta, current_time

def get_heading(compass):
    """Get normalized heading from compass in radians"""
    compass_data = compass.getValues()
    heading = math.atan2(compass_data[0], compass_data[1])
    return heading

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return math.atan2(math.sin(angle), math.cos(angle))

def update_odometry(vL, vR, dt, pose_x, pose_y, pose_theta, AXLE_LENGTH):
    """Update odometry using improved integration"""
    # Simple case - straight line motion
    if abs(vL - vR) < 1e-6:
        pose_x += vL * dt * math.cos(pose_theta)
        pose_y += vL * dt * math.sin(pose_theta)
    else:
        # Standard differential drive kinematics
        v = (vL + vR) / 2.0
        w = (vR - vL) / AXLE_LENGTH
        
        # Use midpoint integration for better accuracy
        theta_mid = pose_theta + 0.5 * w * dt
        pose_x += v * dt * math.cos(theta_mid)
        pose_y += v * dt * math.sin(theta_mid)
        pose_theta += w * dt
    
    return pose_x, pose_y, pose_theta

# Export the main function
export = {
    "main": main,
}
