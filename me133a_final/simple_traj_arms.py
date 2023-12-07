'''simople_traj.py

   Intended to make Atlas_v5 urdf move arms up and down as if doing a pullup.
   Not using back joints at all.

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import QuaternionStamped

from scipy.spatial.transform import Rotation as R

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from me133a_final.utils.GeneratorNode      import GeneratorNode
from me133a_final.utils.TrajectoryUtils    import *
from me133a_final.utils.KinematicChain     import KinematicChain
from me133a_final.utils.TransformHelpers   import *

# list the joints
joint_list = ['back_bkz', 'back_bky', 'back_bkx',
              'l_arm_shz', 'l_arm_shx',
              'l_arm_ely', 'l_arm_elx',
              'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2',
              'l_arm_tip',
              'l_leg_akx', 'l_leg_aky',
              'l_leg_hpx', 'l_leg_hpy', 'l_leg_hpz',
              'l_leg_kny',
              'neck_ry',
              'r_arm_shz', 'r_arm_shx',
              'r_arm_ely', 'r_arm_elx',
              'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2',
              'r_arm_tip',
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
z8 = np.zeros(8).reshape(-1,1)
z13 = np.zeros(13).reshape(-1,1)

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain_larm = KinematicChain(node, 'utorso', 'l_hand_tip', self.jointnames())
        self.chain_rarm = KinematicChain(node, 'utorso', 'r_hand_tip', self.jointnames())

        # Set up the bar angle array subscriber
        self.bar_angle_sub = node.create_subscription(Float64MultiArray, '/bar_angles', self.readBar, 10)
        #self.bar_quat_sub = node.create_subscription(QuaternionStamped, '/bar_quat', self.readBar, 10)

        # Define the various points and initialize as initial joint/task arrays.
        self.L = 0.688 # distance between hands
        
        self.q_l = np.array([-1.201, 0.654,
                             0.0, 1.044,
                             -0.375, 0.467, 0.0]).reshape((-1,1)) # list of joint coords chaining from utorso to lhand
        self.x0_l = np.array([0.42208, self.L/2, 0.92362])
        self.R0_l = R.from_quat([0.62986, -0.6158, -0.34924, 0.31952]).as_matrix()
        (self.x_l, self.R_l, _, _) = self.chain_larm.fkin(self.q_l)
        self.T_l = T_from_Rp(self.R_l, self.x_l)
        
        self.q_r = np.array([1.201, -0.654,
                             0.0, -1.044, 
                             -0.375, -0.467, 0.0]).reshape((-1,1)) # list of joint coords chaining from utorso to rhand
        self.x0_r = np.array([0.42208,-self.L/2, 0.92362]).reshape(-1,1)
        self.R0_r = R.from_quat([0.62986, 0.61584, -0.34923, -0.31945]).as_matrix()
        (self.x_r, self.R_r, _, _) = self.chain_rarm.fkin(self.q_r)
        self.T_r = T_from_Rp(self.R_r, self.x_r)

        # Initialize the bar angles as 0 (standard config)
        self.pan= 0.0
        self.tilt = 0.0

        self.lam = 20.0

    def readBar(self, bar_angles_msg):
        # read in and save published bar pan/tilt angles
        bar_angles = bar_angles_msg.data # unpack ROS msg
        [self.pan, self.tilt] = bar_angles

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names for atlas_v5.urdf
        return joint_list

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        sz = 0.225*np.cos(np.pi*(t)) # path variable for z
        
        # compute additional offset needed from bar tilt to keep hands on bar
        tilt_compensate = self.L/2 * np.sin(self.tilt)

        # desired trajectory of left hand is moving up and down at constant (x,y)
        # TODO: consider tilt angle in z-trajectory of hand
        xd_l = np.array([float(self.x0_l[0]), float(self.x0_l[1]), sz + 0.563 + tilt_compensate]).reshape(-1,1)
        vd_l = np.array([0.0, 0.0, -0.225*np.pi*np.sin(np.pi*(t))]).reshape(-1,1)
        Rd_l = self.R0_l
        wd_l = np.array([0.0,0.0,0.0]).reshape(-1,1)

        # desired trajectory of right hand is moving up and down at constant (x,y)
        # TODO: consider tilt angle in z-trajectory of hand
        xd_r = np.array([float(self.x0_r[0]), float(self.x0_r[1]), sz + 0.563 - tilt_compensate]).reshape(-1,1)
        vd_r = np.array([0.0, 0.0, -0.225*np.pi*np.sin(np.pi*(t))]).reshape(-1,1)
        Rd_r = self.R0_r
        wd_r = np.array([0.0,0.0,0.0]).reshape(-1,1)

        # ikin
        (self.x_l, self.R_l, Jv_l, Jw_l) = self.chain_larm.fkin(self.q_l)
        self.T_l = T_from_Rp(self.R_l, self.x_l)
        e_l = np.vstack((ep(xd_l, self.x_l), eR(Rd_l, self.R_l)))
        J_l = np.vstack((Jv_l, Jw_l))
        xdotd_l = np.vstack((vd_l, wd_l))
        
        (self.x_r, self.R_r, Jv_r, Jw_r) = self.chain_rarm.fkin(self.q_r)
        self.T_r = T_from_Rp(self.R_r, self.x_r)
        e_r = np.vstack((ep(xd_r, self.x_r), eR(Rd_r, self.R_r)))
        J_r = np.vstack((Jv_r, Jw_r))
        xdotd_r = np.vstack((vd_r, wd_r))

        e = np.vstack((e_l, e_r))
        J = np.vstack((np.hstack((J_l, np.zeros((6,7)))),np.hstack((np.zeros((6,7)), J_r)))) # see onenote
        xdotd = np.vstack((xdotd_l, xdotd_r))

        # Compute position/orientation of the pelvis (w.r.t. world).
        xavg = (self.x_l + self.x_r) / 2
        norm2 = np.sqrt(xavg[0]**2 + xavg[1]**2)
        pan_compensate = np.array([norm2*np.cos(self.pan),
                                   norm2*np.sin(self.pan),
                                   xavg[2]])
        ppelvis = pxyz(0.0, 0.0, 1.75) - pan_compensate
        Rpelvis = Rotz(self.pan)
        TPELVIS = T_from_Rp(Rpelvis, ppelvis)
        
        Jwinv = np.transpose(J)@np.linalg.inv(J@np.transpose(J) + (0.05**2)*np.eye(12)) # implement weighted inverse to help near singularities

        qdot = Jwinv@(xdotd + e*self.lam)
        qdot_l = qdot[0:7]
        qdot_r = qdot[7:]
        self.q_l = self.q_l + qdot_l*dt
        self.q_r = self.q_r + qdot_r*dt
        
        q = np.vstack((z3, self.q_l, z8, self.q_r, z13)) # important: this q should be same order as joints presented in joint_list above
        qdot = np.vstack((z3, qdot_l, z8, qdot_r, z13))
        # Return the position and velocity as python lists.
        return (q.flatten().tolist(), qdot.flatten().tolist(), TPELVIS, self.T_l, self.T_r)


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
