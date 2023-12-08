'''This will initialize a bar between atlas's hands, with pan/tilt GUI sliders.

Node: /bar
Publish: /bar_angles          std_msgs/Float64MultiArray
'''
import rclpy
import numpy as np
from me133a_final.utils.BarNodeFloats import *

#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the GUI node (10Hz).
    rclpy.init(args=args)
    bar_angles = [0.0, 0.0, 0.688]
    node = GUINode('bar', bar_angles, 100) # fixme with correct initial orientation

    # Run until interrupted.
    node.run()

    # Shutdown the node and ROS.
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()