import math

def get_heading(compass):
    """Get normalized heading from compass in radians."""
    data = compass.getValues()
    # Calculate heading from compass x and z
    heading = math.atan2(data[0], data[2])
    return heading

def normalize_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# taken from lab5_controller.py, but written by our group for that lab, and slightly modified for this lab
def rotateToAngle(target_angle, robot, compass, timestep, robot_parts, vL, vR, dt, x, y, theta, AXLE_LENGTH):
    """ Rotate to a target angle using compass heading. Also handles odometry updates, and returns the updated odometry values."""
    # Convert to radians for the calculation
    target_angle = math.radians(target_angle)
    
    # Rotation controller parameters
    k_rotation = 3.0  # Proportional gain for rotation
    min_rotation_speed = 1.0  # Minimum speed to overcome friction
    rotation_complete = False
    
    # Rotation loop until target is reached
    while robot.step(timestep) != -1 and not rotation_complete:

        # Yes this is a mess, but it works...
        # Get updated pose based on compass
        compass_values = compass.getValues()
        north_x = compass_values[0]
        north_z = compass_values[1]
        compass_based_angle_raw = math.atan2(-north_x, north_z) - math.pi / 2
        compass_based_angle = (compass_based_angle_raw + math.pi) % (2 * math.pi) - math.pi
        pose_theta_raw = -compass_based_angle - math.pi / 2
        pose_theta = (pose_theta_raw + math.pi) % (2 * math.pi) - math.pi
        # Calculate heading error
        angle_error = normalize_angle(target_angle - pose_theta)
        # Update odometry based on current velocities
        x, y, theta = update_odometry(vL, vR, dt, x, y, pose_theta, AXLE_LENGTH)
        
        # Stop if within tolerance
        if abs(angle_error) < math.radians(3):  # 3 degree tolerance
            vL = 0
            vR = 0
            robot_parts["wheel_left_joint"].setVelocity(vL)
            robot_parts["wheel_right_joint"].setVelocity(vR)
            rotation_complete = True
            break
        
        # Compute rotation speed
        rotation_speed = k_rotation * angle_error
        
        # Apply minimum rotation speed if needed
        if 0 < rotation_speed < min_rotation_speed:
            rotation_speed = min_rotation_speed
        elif -min_rotation_speed < rotation_speed < 0:
            rotation_speed = -min_rotation_speed
        
        # Cap max speed
        max_rotation_speed = 3.5
        if abs(rotation_speed) > max_rotation_speed:
            rotation_speed = max_rotation_speed * (1 if rotation_speed > 0 else -1)
        
        # Set wheel speeds for rotation
        vL = -rotation_speed
        vR = rotation_speed
        
        # Apply wheel velocities
        robot_parts["wheel_left_joint"].setVelocity(vL)
        robot_parts["wheel_right_joint"].setVelocity(vR)
    
    # Make sure robot is stopped after rotation
    vL = 0
    vR = 0
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    # Short settling period
    print("Letting robot settle after rotation...")
    settling_timer = 0
    while settling_timer < 20 and robot.step(timestep) != -1:
        settling_timer += 1
        robot_parts["wheel_left_joint"].setVelocity(0)
        robot_parts["wheel_right_joint"].setVelocity(0)

    return x, y, theta

def update_odometry(vL, vR, dt, x, y, theta, AXLE_LENGTH):
    """Update odometry using midpoint integration for differential drive."""
    if abs(vL - vR) < 1e-6:
        # Robot moving straight
        x += vL * dt * math.cos(theta)
        y += vL * dt * math.sin(theta)
        theta_new = theta
    else:
        # Robot rotating
        v = 0.5 * (vL + vR)
        w = (vR - vL) / AXLE_LENGTH
        # Heading
        theta_new = normalize_angle(theta + w * dt)
        # Midpoint angle
        theta_mid = normalize_angle((theta + theta_new) * 0.5)
        # Position
        x += v * dt * math.cos(theta_mid)
        y += v * dt * math.sin(theta_mid)
    return x, y, theta_new

def main(robot, compass, AXLE_LENGTH, gps, display, vL, vR, prev_time, pose_x, pose_y, pose_theta):
    current_time = robot.getTime()
    # Skip update if first frame
    if prev_time is None:
        return pose_x, pose_y, pose_theta, current_time

    # Compute and clamp dt
    dt = current_time - prev_time
    dt = max(0.0, min(dt, 0.1))
    if dt == 0.0:
        return pose_x, pose_y, pose_theta, current_time

    # Guard against invalid axle length
    if abs(AXLE_LENGTH) < 1e-6:
        AXLE_LENGTH = 0.1

    # Read sensors
    gps_pos = gps.getValues()
    compass_heading = get_heading(compass)

    # Update pose using odometry
    pose_x, pose_y, pose_theta = update_odometry(
        vL, vR, dt, pose_x, pose_y, pose_theta, AXLE_LENGTH)

    # Compass correction
    heading_error = normalize_angle(compass_heading - pose_theta)
    pose_theta = normalize_angle(pose_theta + 0.05 * heading_error)

    # Periodic GPS fusion every 10s (single shot)
    if int(current_time) % 100 == 0 and not getattr(main, '_gps_corrected', False):
        pose_x = 0.9 * pose_x + 0.1 * gps_pos[0]
        pose_y = 0.9 * pose_y + 0.1 * gps_pos[2]
        main._gps_corrected = True
    elif int(current_time) % 100 != 0:
        main._gps_corrected = False

    # Debug output
    print(f"GPS: ({gps_pos[0]:.2f}, {gps_pos[1]:.2f}), "
          f"Heading: {compass_heading:.2f}, "
          f"Odometry: (x: {pose_x:.2f}, y: {pose_y:.2f}, θ: {pose_theta:.2f})")
    
    #gps_x,gps_y,odom_x,odom_y,heading,theta
    #print(f"{gps_pos[0]:.2f},{gps_pos[2]:.2f},{pose_x:.2f},{pose_y:.2f},{compass_heading:.2f},{pose_theta:.2f}")

    return pose_x, pose_y, pose_theta, current_time

# Export API
export = {
    "main": main
}
