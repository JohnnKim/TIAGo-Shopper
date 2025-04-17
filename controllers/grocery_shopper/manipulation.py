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
import pygame # Added for controller input
import sys # Added for controller input error handling
# Remove global prev_position if not used
# global prev_position
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
        # --- Enable Webots Keyboard ---
        keyboard = robot.getKeyboard()
        keyboard.enable(timestep)
        print("Keyboard enabled.")

        # --- Try to Initialize Pygame Joystick ---
        joystick = None
        joystick_enabled = False
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                joystick = pygame.joystick.Joystick(0)
                joystick.init()
                if joystick:
                    print(f"Initialized Joystick: {joystick.get_name()}")
                    print(f"Axes: {joystick.get_numaxes()}, Buttons: {joystick.get_numbuttons()}, Hats: {joystick.get_numhats()}")
                    joystick_enabled = True
                else:
                    print("Warning: Pygame joystick found but failed to initialize.")
            else:
                print("Info: No joystick detected. Using keyboard controls.")
        except Exception as e:
            print(f"Warning: Pygame/Joystick initialization failed: {e}. Using keyboard controls.")
            if pygame.get_init(): # Clean up if pygame initialized partially
                 pygame.quit()
            joystick_enabled = False

        # Control constants
        DEADZONE = 0.1
        AXIS_SENSITIVITY = 0.005 # Smaller value = stiffer control for joystick
        Z_STEP = 0.01 # Step size for Z movement
        KEY_SENSITIVITY = 0.02 # Step size for keyboard movement

        # Get the initial position of the motors
        looping = True
        current_position = [0.5, 0.021, 0.68]  # Forward, centered, slightly above floor

        while robot.step(timestep) != -1 and looping:
            # --- Reset Action Flags ---
            delta_x, delta_y, delta_z = 0.0, 0.0, 0.0
            open_gripper_action = False
            close_gripper_action = False
            move_torso_up = False
            move_torso_down = False
            reset_action = False
            exit_action = False

            # --- Input Reading (Conditional) ---
            if joystick_enabled:
                # Process Pygame events (essential for joystick)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: # Allow closing external window if one exists
                        exit_action = True
                    # Optional: Handle joystick disconnection/reconnection

                # Check connection (basic)
                if not pygame.joystick.get_count() > 0:
                     print("Error: Joystick disconnected during use.", file=sys.stderr)
                     exit_action = True # Or could attempt fallback to keyboard

                if not exit_action:
                    # Read Joystick Axes
                    axis0 = joystick.get_axis(0) # Left Stick X: Left (-1) to Right (+1)
                    axis1 = joystick.get_axis(1) # Left Stick Y: Up (-1) to Down (+1)
                    axis4 = joystick.get_axis(4) # L2 Trigger
                    axis5 = joystick.get_axis(5) # R2 Trigger

                    # Read Joystick Buttons
                    if joystick.get_button(1): exit_action = True      # Circle (Exit)
                    if joystick.get_button(8): open_gripper_action = True # Open Grip
                    if joystick.get_button(9): close_gripper_action = True # Close Grip
                    if joystick.get_button(11): delta_z += Z_STEP        # Move Hand Up
                    if joystick.get_button(12): delta_z -= Z_STEP        # Move Hand Down
                    if joystick.get_button(15): reset_action = True     # Touchpad (Reset)

                    # Apply Joystick Axis Input
                    if abs(axis0) > DEADZONE:
                        delta_y -= axis0 * AXIS_SENSITIVITY # Left/Right (Y)
                    if abs(axis1) > DEADZONE:
                        delta_x -= axis1 * AXIS_SENSITIVITY # Forward/Back (X)

                    # Apply Trigger Input
                    trigger_threshold = -0.9
                    if axis5 > trigger_threshold: move_torso_up = True # R2
                    if axis4 > trigger_threshold: move_torso_down = True # L2

            else: # Keyboard Control
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

            # --- Apply Actions ---
            if exit_action:
                looping = False
                break # Exit the loop immediately

            # Update position based on inputs
            current_position[0] += delta_x
            current_position[1] += delta_y
            current_position[2] += delta_z

            # Apply discrete actions
            if open_gripper_action: openGrip()
            if close_gripper_action: closeGrip()
            if move_torso_up: moveUp()
            if move_torso_down: moveDown()
            if reset_action:
                current_position = [0.5, 0.021, 0.68] # Reset to origin
                print("Position Reset")


            # --- Robot Control ---
            # Stop wheels (redundant?)
            robot_parts["wheel_left_joint"].setVelocity(0)
            robot_parts["wheel_right_joint"].setVelocity(0)

            # Calculate IK for the target position
            ikResults = calculateIk(current_position, orient=True, orientation_mode="Y")

            # Move the arm
            moveArmToTarget(ikResults)

            # Print current target position
            print("current_position: ", ["{:.3f}".format(p) for p in current_position])

        # --- Cleanup ---
        if joystick_enabled and pygame.get_init(): # Quit pygame only if it was used
            pygame.quit()
        print("Exiting manual mode.")
        # Keyboard is disabled automatically when the controller exits

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
