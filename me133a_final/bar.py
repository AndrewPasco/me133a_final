'''This will initialize a bar between atlas's hands, with pan/tilt GUI sliders.

Node: /bar
Publish: /bar_quat          geometry_msgs/QuaternionStamped
'''
import rclpy
import numpy as np
from me133a_final.utils.shapes import *
from me133a_final.utils.shapes import *

#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the GUI node (10Hz).
    rclpy.init(args=args)
    bar_quat = quat_from_R(Rotx(np.pi/2))
    node = GUINode('bar', bar_quat, 10) # fixme with correct initial orientation

    # Run until interrupted.
    node.run()

    # Shutdown the node and ROS.
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()