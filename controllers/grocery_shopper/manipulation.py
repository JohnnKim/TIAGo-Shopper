import numpy as np
# Patch np.float_
np.float = float
# Patch np.bool_
np.bool = bool
# Patch np.int_
np.int = int
import matplotlib.pyplot as plt
import ikpy.chain
import ikpy.utils.plot as plot_utils #plotting utils, install with " pip install 'ikpy[plot]' "


def main(robot, display, robot_parts, gripper_status, vL, vR, right_gripper_enc, left_gripper_enc):

    # Initialize the chain
    my_chain = ikpy.chain.Chain.from_urdf_file("./robot_urdf.urdf")

    # target_position = [ 0.1, -0.2, 0.1]
    # import matplotlib.pyplot as plt
    # fig, ax = plot_utils.init_3d_figure()
    # my_chain.plot(my_chain.inverse_kinematics(target_position), ax, target=target_position)
    # plt.xlim(-0.1, 0.1)
    # plt.ylim(-0.1, 0.1)
    # plt.show()

    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
