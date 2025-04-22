import math

def get_heading(compass):
    """Get normalized heading from compass in radians."""
    data = compass.getValues()
    # Use Z-axis for forward direction
    heading = math.atan2(data[0], data[2])
    return heading

def normalize_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

def update_odometry(vL, vR, dt, x, y, theta, AXLE_LENGTH):
    """Update odometry using midpoint integration for differential drive."""
    # Straight line approximation
    if abs(vL - vR) < 1e-6:
        x += vL * dt * math.cos(theta)
        y += vL * dt * math.sin(theta)
        theta_new = theta
    else:
        v = 0.5 * (vL + vR)
        w = (vR - vL) / AXLE_LENGTH
        # Integrate heading
        theta_new = normalize_angle(theta + w * dt)
        # Midpoint angle
        theta_mid = normalize_angle((theta + theta_new) * 0.5)
        # Integrate position
        x += v * dt * math.cos(theta_mid)
        y += v * dt * math.sin(theta_mid)
    return x, y, theta_new

def main(robot, compass, AXLE_LENGTH, gps, display, vL, vR, prev_time, pose_x, pose_y, pose_theta):
    current_time = robot.getTime()
    # First‐loop guard
    if prev_time is None:
        return pose_x, pose_y, pose_theta, current_time

    # Compute and clamp dt
    dt = current_time - prev_time
    dt = max(0.0, min(dt, 0.1))
    if dt == 0.0:
        return pose_x, pose_y, pose_theta, current_time

    # Ensure valid axle length
    if abs(AXLE_LENGTH) < 1e-6:
        AXLE_LENGTH = 0.1

    # Read sensors
    gps_pos = gps.getValues()
    compass_heading = get_heading(compass)

    # Pure odometry update
    pose_x, pose_y, pose_theta = update_odometry(
        vL, vR, dt, pose_x, pose_y, pose_theta, AXLE_LENGTH)

    # Compass‐based heading correction
    heading_error = normalize_angle(compass_heading - pose_theta)
    pose_theta = normalize_angle(pose_theta + 0.05 * heading_error)

    # Periodic GPS fusion every 10s (single shot)
    if int(current_time) % 10 == 0 and not getattr(main, '_gps_corrected', False):
        pose_x = 0.9 * pose_x + 0.1 * gps_pos[0]
        pose_y = 0.9 * pose_y + 0.1 * gps_pos[2]
        main._gps_corrected = True
    elif int(current_time) % 10 != 0:
        main._gps_corrected = False

    # Debug output
    # print(f"GPS: ({gps_pos[0]:.2f}, {gps_pos[2]:.2f}), "
    #       f"Heading: {compass_heading:.2f}, "
    #       f"Odometry: (x: {pose_x:.2f}, y: {pose_y:.2f}, θ: {pose_theta:.2f})")
    
    #gps_x,gps_y,odom_x,odom_y,heading,theta
    print(f"{gps_pos[0]:.2f},{gps_pos[2]:.2f},{pose_x:.2f},{pose_y:.2f},{compass_heading:.2f},{pose_theta:.2f}")

    return pose_x, pose_y, pose_theta, current_time

# Export API
export = {
    "main": main
}