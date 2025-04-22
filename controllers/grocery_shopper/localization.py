import numpy as np
import math

"""
Entry point for odometry
"""
def main(robot, compass, AXLE_LENGTH, gps, display, vL, vR, prev_time,
         pose_x, pose_y, pose_theta, last_gps_update_time):
    # Validate parameters
    if abs(AXLE_LENGTH) < 1e-6:
        print("Warning: AXLE_LENGTH is too small, using default value")
        AXLE_LENGTH = 0.1  # Safe fallback
    
    # Process timing
    current_time = robot.getTime()
    dt = current_time - prev_time
    if dt < 1e-6:
        return pose_x, pose_y, pose_theta, current_time, last_gps_update_time

    # Sensor readings
    gps_position = gps.getValues()
    compass_heading = get_heading(compass)  # Corrected axis
    
    # Update odometry
    pose_x, pose_y, pose_theta = update_odometry(vL, vR, dt, pose_x, pose_y, pose_theta, AXLE_LENGTH)
    pose_theta = normalize_angle(pose_theta)
    
    # Sensor fusion
    if abs(vL) > 0.01 or abs(vR) > 0.01:
        # Compass correction (gentle)
        heading_error = normalize_angle(compass_heading - pose_theta)
        pose_theta += 0.01 * heading_error  # Reduced gain
        
        # Time-based GPS correction every ~10 seconds
        if current_time - last_gps_update_time >= 10.0:
            pose_x = 0.9 * pose_x + 0.1 * gps_position[0]
            pose_y = 0.9 * pose_y + 0.1 * gps_position[2]
            last_gps_update_time = current_time

    # Debug output
    print(f"GPS: ({gps_position[0]:.2f}, {gps_position[2]:.2f}), "
          f"Heading: {compass_heading:.2f}, "
          f"Odometry: (x: {pose_x:.2f}, y: {pose_y:.2f}, θ: {pose_theta:.2f})")
    
    return pose_x, pose_y, pose_theta, current_time, last_gps_update_time


def get_heading(compass):
    """Get normalized heading from compass in radians"""
    compass_data = compass.getValues()
    return math.atan2(compass_data[0], compass_data[2])  # FIXED AXIS


def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return math.atan2(math.sin(angle), math.cos(angle))


def update_odometry(vL, vR, dt, pose_x, pose_y, pose_theta, AXLE_LENGTH):
    """Update odometry using midpoint integration"""
    if abs(vL - vR) < 1e-6:
        pose_x += vL * dt * math.cos(pose_theta)
        pose_y += vL * dt * math.sin(pose_theta)
    else:
        v = (vL + vR) / 2.0
        w = (vR - vL) / AXLE_LENGTH
        theta_mid = pose_theta + 0.5 * w * dt
        pose_x += v * dt * math.cos(theta_mid)
        pose_y += v * dt * math.sin(theta_mid)
        pose_theta += w * dt

    return pose_x, pose_y, pose_theta


# Export for compatibility
export = {
    "main": main,
}
