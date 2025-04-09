"""
This file contains the manipulation module for the grocery shopper.
It contains the functions for the arm to move to a target position and for the gripper to open and close.
All functions are written by Mitch Siegel
"""

import numpy as np
# patching for ikpy cause its old
np.float = float
np.bool = bool
np.int = int
import math
import matplotlib.pyplot as plt
import ikpy.chain
import ikpy.utils.plot as plot_utils #plotting utils, install with " pip install 'ikpy[plot]' "

# ---- HELPER FUNCTIONS ----

def get_gripper_status(right_gripper_enc, left_gripper_enc):
    if right_gripper_enc.getValue()<=0.005:
        return "closed"
    else:
        return "open"

def toggle_gripper(robot_parts, right_gripper_enc, left_gripper_enc):
    if get_gripper_status(right_gripper_enc, left_gripper_enc)=="open":
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
    else:
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)

# Initialize the chain
base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]
my_chain = ikpy.chain.Chain.from_urdf_file("robot_urdf.urdf", base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"])

# --- Manually control arm ---


            

# ---- MAIN FUNCTION ----
def main(robot, display, robot_parts, gripper_status, right_gripper_enc, left_gripper_enc, timestep, x,y,z):
 
    part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
                "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
                "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

    for link_id in range(len(my_chain.links)):

        # This is the actual link object
        link = my_chain.links[link_id]
        
        # I've disabled "torso_lift_joint" manually as it can cause
        # the TIAGO to become unstable.
        if link.name not in part_names or  link.name =="torso_lift_joint":
            print("Disabling {}".format(link.name))
            my_chain.active_links_mask[link_id] = False
            
    # Initialize the arm motors and encoders.
    motors = []
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

    def checkArmAtPosition(ikResults, cutoff=0.00005):
        '''Checks if arm at position, given ikResults'''
        
        # Get the initial position of the motors
        initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]

        # Calculate the arm
        arm_error = 0
        for item in range(14):
            arm_error += (initial_position[item] - ikResults[item])**2
        arm_error = math.sqrt(arm_error)

        if arm_error < cutoff:
            print("Arm at position.")
            return True
        return False

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
        position = my_chain.forward_kinematics(ikResults)
        squared_distance = math.sqrt(
            (position[0, 3] - offset_target[0])**2 + 
            (position[1, 3] - offset_target[1])**2 + 
            (position[2, 3] - offset_target[2])**2
        )
        print(f"IK calculated with error - {squared_distance}")
        
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
        keyboard = robot.getKeyboard()
        keyboard.enable(timestep)

        # Get the initial position of the motors
        looping = True
        current_position = [0.5, 0.021, 0.68]  # Forward, centered, slightly above floor # Forward, centered, slightly above floor
        while robot.step(timestep) != -1 and looping:
            # stop the wheels
            robot_parts["wheel_left_joint"].setVelocity(0)
            robot_parts["wheel_right_joint"].setVelocity(0)
            key = robot.getKeyboard().getKey()
            
            if key == 65:  # A key - Move Left
                current_position[1] += 0.02
            elif key == 68: # D key - Move Right
                current_position[1] -= 0.02
            elif key == 87: # W key - Move Forward
                current_position[0] += 0.02 
            elif key == 83: # S key - Move Backward
                current_position[0] -= 0.02
            elif key == 81: # Q key - Move Up
                current_position[2] += 0.02 
            elif key == 69: # E key - Move Down
                current_position[2] -= 0.02
            elif key == 71: # G key - Exit manual mode (and close gripper)
                closeGrip()
            elif key == 72: # H key - Open Gripper
                openGrip()
            elif key == 82: # R key - Move Up
                moveUp()
            elif key == 70: # F key - Move Down
                moveDown()
            elif key == 67: # C key - Exit manual mode
                looping = False
            elif key == 66: # B key - Move Backward
                current_position = [0.5, 0.021, 0.68] #hard reset to origin

            # Only calculate IK and move arm if position was changed

            # Calculate IK for this position
            ikResults = calculateIk(current_position, orient=True, orientation_mode="Y")
                        
            # Move the arm to the target position
            moveArmToTarget(ikResults)

            # Check if the arm is at the target position
            print("current_position: ", current_position)

    #x, y, z = abs(x), abs(y), abs(z)
    
    manual_mode(robot, robot_parts, timestep)

    # test moving the arm to a fixed position
    #the robot at this point is at GPS [-2.75167,1.17914, 0.0920304]
    # current_position = [0.92, -0.139, 0.16]
    # arm_at_target = False
    # while not arm_at_target and robot.step(timestep) != -1:
    #     robot_parts["wheel_left_joint"].setVelocity(0)
    #     robot_parts["wheel_right_joint"].setVelocity(0)
    #     ikResults = calculateIk(current_position, orient=True, orientation_mode="Y")
    #     moveArmToTarget(ikResults)
    #     if checkArmAtPosition(ikResults):
    #         print("Arm at target")
    #         arm_at_target = True


    return
    arm_at_target = False
    while robot.step(timestep) != -1 and not arm_at_target:
        robot_parts["wheel_left_joint"].setVelocity(0)
        robot_parts["wheel_right_joint"].setVelocity(0)
        if not arm_at_target:
            ikResults = calculateIk([x,y,z])
            moveArmToTarget(ikResults)
            if checkArmAtPosition(ikResults):
                print("Arm at target")                
                arm_at_target = True
        else:
            robot_parts["wheel_left_joint"].setVelocity(0)
            robot_parts["wheel_right_joint"].setVelocity(0)


    # plt.xlim(-0.1, 0.1)
    # plt.ylim(-0.1, 0.1)
    # plt.show()
