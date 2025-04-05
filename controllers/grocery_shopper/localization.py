import numpy as np
import math

# Map Initialization
map = np.zeros((360, 360))  # TODO: Validate map size based on environment scale


""" 
Entry point for odometry
"""
def main(robot, compass, AXLE_LENGTH, gps, display, vL, vR, prev_time, pose_x, pose_y, pose_theta):
    # Implemented April 2 - John
    # Function to get heading from compass
    def get_heading():
        compass_data = compass.getValues()
        heading = math.atan2(compass_data[0], compass_data[1])
        return heading

    # Implemented April 2 - John
    # Function to update odometry from wheel velocities
    def update_odometry(vL, vR, dt, pose_x, pose_y, pose_theta):
        v = (vL + vR) / 2.0
        w = (vR - vL) / AXLE_LENGTH
        pose_theta += w * dt
        pose_x += v * dt * math.cos(pose_theta)
        pose_y += v * dt * math.sin(pose_theta)
        return pose_x, pose_y, pose_theta

    # ------------------------------------------------------------------
    # Helper Functions

    # Implemented April 2 - John
    # Compute time step
    current_time = robot.getTime()
    dt = current_time - prev_time
    prev_time = current_time

    # Read GPS data
    gps_position = gps.getValues()
    
    # Read Compass data
    heading = get_heading()
    
    # Update odometry
    pose_x, pose_y, pose_theta = update_odometry(vL, vR, dt, pose_x, pose_y, pose_theta)
    
    # Print localization info for debugging
    print(f"GPS: {gps_position}, Heading: {heading:.2f}, Odometry: (x: {pose_x:.2f}, y: {pose_y:.2f}, θ: {pose_theta:.2f})")
    
    # TODO: Store GPS and odometry data for debugging
    # TODO: Validate map updates with test cases
    # TODO: Later - implement Kalman Filter for fusion

    return pose_x, pose_y, pose_theta, prev_time

export = {
    "main": main,

}
