"""Launch me133a_final Atlas simple trajectory with only arms

   ros2 launch me133a_final simple_traj_arms.launch.py

This should start
  1) RVIZ, ready to view the robot
  2) The robot_state_publisher to broadcast the robot model
  3) The simple starting trajectory code to move the joints
  4) The bar node and GUI

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import DeclareLaunchArgument
from launch.actions                    import OpaqueFunction
from launch.actions                    import Shutdown
from launch.substitutions              import LaunchConfiguration
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('me133a_final'), 'rviz/viewurdf.rviz')

    # Locate the URDF file.
    urdf = os.path.join(pkgdir('atlas_description'), 'urdf/atlas_v5.urdf')

    # Load the robot's URDF file (XML).
    with open(urdf, 'r') as file:
        robot_description = file.read()


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the robot_state_publisher.
    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    # Configure a node for RVIZ
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())
    
    # Configure a node for the joint trajectory
    node_trajectory = Node(
        name       = 'trajectory', 
        package    = 'me133a_final',
        executable = 'simple_traj_arms',
        output     = 'screen')
    
    # Configure a node for the point_publisher.
    node_bar = Node(
        name       = 'bar',
        package    = 'me133a_final',
        executable = 'bar',
        output     = 'screen',
        on_exit    = Shutdown())
    

    ######################################################################
    # RETURN THE ELEMENTS IN ONE LIST
    return LaunchDescription([
        # Start the robot_state_publisher, RVIZ, and the trajectory.
        node_robot_state_publisher,
        node_rviz,
        node_trajectory,
        node_bar,
    ])
