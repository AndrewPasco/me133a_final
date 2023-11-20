'''simople_traj.py

   Intended to make Atlas_v5 urdf move arms up and down as if doing a pullup.

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from std_msgs.msg import Float64

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain

# list the joints in same order as atlas_v5.urdf
joint_list = ['back_bkx', 'back_bky', 'back_bkz',
              'l_arm_elx', 'l_arm_ely',
              'l_arm_shx', 'l_arm_shz',
              'l_arm_wrx', 'l_arm_wry', 'l_arm_wry2',
              'l_leg_akx', 'l_leg_aky',
              'l_leg_hpx', 'l_leg_hpy', 'l_leg_hpz',
              'l_leg_kny',
              'neck_ry',
              'r_arm_elx', 'r_arm_ely',
              'r_arm_shx', 'r_arm_shz',
              'r_arm_wrx', 'r_arm_wry', 'r_arm_wry2',
              'r_leg_akx', 'r_leg_aky',
              'r_leg_hpx', 'r_leg_hpy', 'r_leg_hpz',
              'r_leg_kny',
              'r_situational_awareness_camera_joint',
              'r_situational_awareness_camera_optical_frame_joint',
              'l_situational_awareness_camera_joint',
              'l_situational_awareness_camera_optical_frame_joint',
              'rear_situational_awareness_camera_joint',
              'rear_situational_awareness_camera_optical_frame_joint']
'''
# joints listed in order needed for kinematic chain?
joint_list_rhand = ['back_bkz', 'back_bky', 'back_bkx',
                   'r_arm_shz', 'r_arm_shx',
                   'r_arm_ely', 'r_arm_elx',
                   'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2']

joint_list_lhand = ['back_bkz', 'back_bky', 'back_bkx',
                   'l_arm_shz', 'l_arm_shx',
                   'l_arm_ely', 'l_arm_elx',
                   'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2']

joint_list_rfoot = []

joint_list_lfoot = []

jointnames_dict = {'rhand':joint_list_rhand,
                   'lhand':joint_list_lhand,
                   'rfoot':joint_list_rfoot,
                   'lfoot':joint_list_lfoot}
'''
num_joints = 36 # total number of atlas joints

z7 = np.zeros(7).reshape(-1,1)
z12 = np.zeros(12).reshape(-1,1)

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain_rarm = KinematicChain(node, 'pelvis', 'r_hand', self.jointnames())
        self.chain_larm = KinematicChain(node, 'pelvis', 'l_hand', self.jointnames())

        # Set up the condition number publisher
        self.pub = node.create_publisher(Float64, '/condition', 10)

        # Define the various points and initialize as initial joint/task arrays.
        self.q_r = np.array([0.0, 0.0, 0.0,
                             -0.662, 0.0,
                             -1.027, 0.743,
                             0.0, -0.505, 0.0]).reshape((-1,1))
        (self.x_r, self.R_r, _, _) = self.chain_rarm.fkin(self.q_r)
        
        self.q_l = np.array([0.0, 0.0, 0.0,
                             0.662, 0.0,
                             1.027, -0.743,
                             0.0, 0.505, 0.0]).reshape((-1,1))
        (self.x_l, self.R_l, _, _) = self.chain_larm.fkin(self.q_l)

        self.lam = 20.0

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names for atlas_v5.urdf
        return joint_list

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt): # currently just outputs q0 pose, no changes
        '''
        pd = np.array([0.0, 0.95 - 0.25*np.cos(t), 0.60 + 0.25*sin(t)]).reshape(-1,1)
        vd = np.array([0.0, 0.25*np.sin(t), 0.25*np.cos(t)]).reshape(-1,1)
        Rd = Reye()
        wd = np.array([0.0,0.0,0.0]).reshape(-1,1)

        # ikin
        (self.x, self.R, Jv, Jw) = self.chain.fkin(self.q)
        e = np.vstack((ep(pd, self.x), eR(Rd, self.R)))
        J = np.vstack((Jv, Jw))
        xdotd = np.vstack((vd, wd))
        Jwinv = np.transpose(J)@np.linalg.inv(J@np.transpose(J) + (0.2**2)*np.eye(6))
        qdots = self.lam2*np.array([0.0,0.0,0.0,-np.pi/2 - float(self.q[3]),0.0,0.0,0.0]).reshape(-1,1)
        qdot = Jwinv@(xdotd + e*self.lam) + (np.eye(7) - Jwinv@J)@qdots
        self.q = self.q + qdot*dt'''
        q = np.vstack((self.q_l, z7, self.q_r[3:], z12))
        # Return the position and velocity as python lists.
        return (q.flatten().tolist(), np.zeros(num_joints).reshape(-1,1).flatten().tolist())


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
