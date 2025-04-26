"""
This file contains the manipulation module for the grocery shopper.
It contains the functions for the arm to move to a target position and for the gripper to open and close.
All functions are written by Mitch Siegel, unless otherwise noted.
"""

import numpy as np
# patching for ikpy cause its old
np.float = float
np.bool = bool
np.int = int
import ikpy.chain

base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]
my_chain = ikpy.chain.Chain.from_urdf_file("robot_urdf.urdf", base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"])
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
                "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
                "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")
motors = []


# Code was written here by Mitch Siegel. Gemini 2.5 pro assisted with debugging and writing comments/code where noted.
def main(robot, robot_parts, timestep):
    """
    Main function for the manipulation code.
    Call this function from main to start the manual IK control loop.
    """
    global my_chain, motors, part_names
    for link_id in range(len(my_chain.links)):

        # This is the actual link object
        link = my_chain.links[link_id]
        
        if link.name not in part_names or  link.name =="torso_lift_joint":
            print("Disabling {}".format(link.name))
            my_chain.active_links_mask[link_id] = False
        
    # Initialize the arm motors and encoders.
    for link in my_chain.links:
        if link.name in part_names and link.name != "torso_lift_joint":
            motor = robot.getDevice(link.name)

            # Make sure to account for any motors that
            # require a different maximum velocity!
            if link.name == "torso_lift_joint":
                motor.setVelocity(0.07)
            else:
                motor.setVelocity(1)
                
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(timestep)
            motors.append(motor)

    # nested helper functions
    def moveArmToTarget(ikResults):
        '''Moves arm given ikResults'''
        # Set the robot motors
        for res in range(len(ikResults)):
            # Only set position if the link is part of robot_parts AND active in the IK chain
            if my_chain.links[res].name in part_names and my_chain.active_links_mask[res]:
                robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])

    def calculateIk(offset_target, orient=True, orientation_mode="Y", target_orientation=[0,0,1]):
        '''
        This will calculate the iK given a target in robot coords
        Parameters
        ----------
        param offset_target: a vector specifying the target position of the end effector
        param orient: whether or not to orient, default True
        param orientation_mode: either "X", "Y", or "Z", default "Y"
        param target_orientation: the target orientation vector, default [0,0,1]

        Returns
        ----------
        rtype: bool
            returns: whether or not the arm is at the target
        '''
        
        # Get the number of links in the chain
        num_links = len(my_chain.links)
        
        # Create initial position array with the correct size
        initial_position = [0] * num_links
    
        # Map each motor to its corresponding link position
        motor_idx = 0
        for i in range(num_links):
            link_name = my_chain.links[i].name
            if link_name in part_names and link_name != "torso_lift_joint":
                if motor_idx < len(motors):
                    initial_position[i] = motors[motor_idx].getPositionSensor().getValue()
                    motor_idx += 1
        
        # Calculate IK
        ikResults = my_chain.inverse_kinematics(
            offset_target, 
            initial_position=initial_position,
            target_orientation=target_orientation, 
            orientation_mode=orientation_mode
        )
        
        # Validate result
        #position = my_chain.forward_kinematics(ikResults)
        #squared_distance = math.sqrt(
        #    (position[0, 3] - offset_target[0])**2 + 
        #    (position[1, 3] - offset_target[1])**2 + 
        #    (position[2, 3] - offset_target[2])**2
        #)
        #print(f"IK calculated with error - {squared_distance}")
        
        return ikResults

    def closeGrip():
        robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
        robot.getDevice("gripper_left_finger_joint").setPosition(0.0)
    def openGrip():
        robot.getDevice("gripper_right_finger_joint").setPosition(0.045)
        robot.getDevice("gripper_left_finger_joint").setPosition(0.045)

    def moveUp():
        robot.getDevice("torso_lift_joint").setPosition(0.35)
    def moveDown():
        robot.getDevice("torso_lift_joint").setPosition(0.0)

    def manual_mode(robot, robot_parts, timestep):
        # --- Enable Webots Keyboard ---
        keyboard = robot.getKeyboard()
        keyboard.enable(timestep)
        print("Keyboard enabled.")

        # Constants for the manual IK control loop, suggested by Gemini 2.5 pro
        Z_STEP = 0.01 # Step size for Z movement (IK target)
        KEY_SENSITIVITY = 0.02 # Step size for keyboard movement (IK target)

        # Get the initial position of the motors for IK
        looping = True
        current_position = [0.5, 0.021, 0.68]  # Forward, centered, slightly above floor (IK Target)


        #print controls to console (generated by Gemini 2.5 pro)
        print("Controls:")
        print("A - Move Left")
        print("D - Move Right")
        print("W - Move Forward")
        print("S - Move Backward")
        print("Q - Move Hand Up")
        print("E - Move Hand Down")
        print("H - Open Gripper")
        print("G - Close Gripper")
        print("R - Move Torso Up")
        print("F - Move Torso Down")
        print("B - Reset Position")
        print("C - Exit Manual Mode")

        # Main loop for the manual IK control loop
        while robot.step(timestep) != -1 and looping:
            # --- Reset Action Flags --- (generated by Gemini 2.5 pro)
            delta_x, delta_y, delta_z = 0.0, 0.0, 0.0
            open_gripper_action = False
            close_gripper_action = False
            move_torso_up = False
            move_torso_down = False
            reset_action = False
            exit_action = False

            key = keyboard.getKey()
            if key == 65: delta_y += KEY_SENSITIVITY   # A key - Move Left
            elif key == 68: delta_y -= KEY_SENSITIVITY  # D key - Move Right
            elif key == 87: delta_x += KEY_SENSITIVITY   # W key - Move Forward
            elif key == 83: delta_x -= KEY_SENSITIVITY  # S key - Move Backward
            elif key == 81: delta_z += Z_STEP           # Q key - Move Hand Up
            elif key == 69: delta_z -= Z_STEP           # E key - Move Hand Down
            elif key == 72: open_gripper_action = True # H key - Open Gripper
            elif key == 71: close_gripper_action = True # G key - Close Gripper (changed from exit)
            elif key == 82: move_torso_up = True       # R key - Move Torso Up
            elif key == 70: move_torso_down = True     # F key - Move Torso Down
            elif key == 66: reset_action = True        # B key - Reset Position (changed from back)
            elif key == 67: exit_action = True         # C key - Exit manual mode

            # --- Apply Actions --- (generated by Gemini 2.5 pro)
            if exit_action:
                looping = False
                break # Exit the loop immediately

            # Update position based on inputs
            current_position[0] += delta_x
            current_position[1] += delta_y
            current_position[2] += delta_z

            # Apply discrete actions (generated by Gemini 2.5 pro)
            if open_gripper_action: openGrip()
            if close_gripper_action: closeGrip()
            if move_torso_up: moveUp()
            if move_torso_down: moveDown()
            if reset_action:
                current_position = [0.5, 0.021, 0.68] #go back to default position
                print("Position Reset")

            robot_parts["wheel_left_joint"].setVelocity(0) #probably my fault but the wheels don't stop unless we force them to
            robot_parts["wheel_right_joint"].setVelocity(0)

            #calculate and move
            ikResults = calculateIk(current_position, orient=True, orientation_mode="Y")
            moveArmToTarget(ikResults)

        print("Exiting manual mode.")

    manual_mode(robot, robot_parts, timestep) #call the manual mode function

    return
    

