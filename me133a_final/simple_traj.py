'''simople_traj.py

   Intended to make Atlas_v5 urdf move arms up and down as if doing a pullup.

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from std_msgs.msg import Float64

from scipy.spatial.transform import Rotation as R

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from KinematicChain    import *

# list the joints
joint_list = ['back_bkz', 'back_bky', 'back_bkx',
              'l_arm_shz', 'l_arm_shx',
              'l_arm_ely', 'l_arm_elx',
              'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2',
              'l_leg_akx', 'l_leg_aky',
              'l_leg_hpx', 'l_leg_hpy', 'l_leg_hpz',
              'l_leg_kny',
              'neck_ry',
              'r_arm_shz', 'r_arm_shx',
              'r_arm_ely', 'r_arm_elx',
              'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2',
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

z3 = np.zeros(3).reshape(-1,1)
z7 = np.zeros(7).reshape(-1,1)
z12 = np.zeros(12).reshape(-1,1)

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain_larm = KinematicChain(node, 'pelvis', 'l_hand', self.jointnames()) # chain from 'utorso' if only want 7dof arm
        self.chain_rarm = KinematicChain(node, 'pelvis', 'r_hand', self.jointnames()) # chain from 'utorso' if only want 7dof arm

        # Set up the condition number publisher
        self.pub = node.create_publisher(Float64, '/condition', 10)

        # Define the various points and initialize as initial joint/task arrays.
        self.q_l = np.array([0.0, 0.0, 0.0,
                             0.662, 0.0,
                             1.027, -0.743,
                             0.0, 0.505, 0.0]).reshape((-1,1)) # list of joint coords chaining from pelvis to hand
        self.x0_l = np.array([0.30835, 0.40821, 0.78949])
        self.R0_l = R.from_quat([0.73478, -0.1082, -0.059397, 0.66698]).as_matrix()
        (self.x_l, self.R_l, _, _) = self.chain_larm.fkin(self.q_l)
        
        self.q_r = np.array([0.0, 0.0, 0.0,
                             -0.662, 0.0,
                             -1.027, 0.743,
                             0.0, -0.505, 0.0]).reshape((-1,1)) # list of joint coords chaining from pelvis to hand
        self.x0_r = np.array([0.30835,-0.40821, 0.78949]).reshape(-1,1)
        self.R0_r = R.from_quat([-0.61432, -0.41739, 0.40747, 0.53138]).as_matrix()
        (self.x_r, self.R_r, _, _) = self.chain_rarm.fkin(self.q_r)

        self.lam = 20.0

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names for atlas_v5.urdf
        return joint_list

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):

        # desired trajectory of left hand is moving up and down at constant (x,y)
        xd_l = np.array([0.30835, 0.40821, 0.2*np.cos(np.pi*(t-0.11)) + 0.6]).reshape(-1,1)
        vd_l = np.array([0.0, 0.0, -0.2*np.pi*np.sin(np.pi*(t-0.11))]).reshape(-1,1)
        Rd_l = self.R0_l
        wd_l = np.array([0.0,0.0,0.0]).reshape(-1,1)

        # desired trajectory of right hand is moving up and down at constant (x,y)
        xd_r = np.array([0.30835, -0.40821, 0.2*np.cos(np.pi*(t-0.11)) + 0.6]).reshape(-1,1)
        vd_r = np.array([0.0, 0.0, -0.2*np.pi*np.sin(np.pi*(t-0.11))]).reshape(-1,1)
        Rd_r = self.R0_r
        wd_r = np.array([0.0,0.0,0.0]).reshape(-1,1)

        # ikin
        (self.x_l, self.R_l, Jv_l, Jw_l) = self.chain_larm.fkin(self.q_l)
        e_l = np.vstack((ep(xd_l, self.x_l), eR(Rd_l, self.R_l)))
        J_l = np.vstack((Jv_l, Jw_l))
        A = J_l[:,0:3]
        B = J_l[:,3:]
        xdotd_l = np.vstack((vd_l, wd_l))
        
        (self.x_r, self.R_r, Jv_r, Jw_r) = self.chain_rarm.fkin(self.q_r)
        e_r = np.vstack((ep(xd_r, self.x_r), eR(Rd_r, self.R_r)))
        J_r = np.vstack((Jv_r, Jw_r))
        C = J_r[:,0:3]
        D = J_r[:,3:]
        xdotd_r = np.vstack((vd_r, wd_r))

        e = np.vstack((e_l, e_r))
        J = np.vstack((np.hstack((A, B, np.zeros((6,7)))),np.hstack((C, np.zeros((6,7)), D)))) # see onenote
        xdotd = np.vstack((xdotd_l, xdotd_r))

        qdot = np.linalg.pinv(J)@(xdotd + e*self.lam)
        qdot_l = qdot[0:10]
        qdot_r = np.concatenate((qdot[0:3], qdot[10:]))
        self.q_l = self.q_l + qdot_l*dt
        self.q_r = self.q_r + qdot_r*dt


        q = np.vstack((self.q_l, z7, self.q_r[3:], z12)) # important: this q should be same order as joints presented in joint_list above
        qdot = np.vstack((qdot_l, z7, qdot_r[3:], z12))
        # Return the position and velocity as python lists.
        return (q.flatten().tolist(), qdot.flatten().tolist())


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
