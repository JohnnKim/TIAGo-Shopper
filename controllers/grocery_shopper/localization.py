import numpy as np
import math
map = np.zeros((360, 360))# Map Initialization (#TODO check size)


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
    def update(vL, vR, dt, pose_x, pose_y, pose_theta):
        v = (vL + vR) / 2.0
        w = (vR - vL) / AXLE_LENGTH
        pose_theta += w * dt
        pose_x += v * dt * math.cos(pose_theta)
        pose_y += v * dt * math.sin(pose_theta)


    # Odometry drift correction (to be expanded)
    # TODO: Refine odometry calculations with wheel encoder data for better accuracy
    # TODO: Implement sensor fusion to correct odometry drift

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
    update(vL, vR, dt, pose_x, pose_y, pose_theta)
    
    # Print localization info for debugging
    print(f"GPS: {gps_position}, Heading: {heading}, Odometry: (x: {pose_x}, y: {pose_y}, theta: {pose_theta})")
    
    # TODO: Store GPS and odometry data for debugging
    # TODO: Validate the map using test cases to make sure accuracy is good




export = {
    "main": main,

}