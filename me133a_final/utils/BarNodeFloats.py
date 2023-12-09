"""shapes.py

   Publish a cylinder "pull up bar", pan/tilt set by two GUI sliders.

   This generates a quaternion message, as well as a visualization marker
   array, which rviz can render.

   Node:        /bar
   Publish:     /bar_angles                       std_msgs/Float64MultiArray
   Publish:     /visualization_marker_array     visualization_msgs/MarkerArray

"""

import rclpy
import numpy as np
import signal
import sys
import threading

from PyQt5.QtCore           import (Qt, QTimer)
from PyQt5.QtWidgets        import (QApplication,
                                    QWidget, QLabel, QHBoxLayout, QVBoxLayout,
                                    QSlider, QCheckBox)

from rclpy.node             import Node
from rclpy.qos              import QoSProfile, DurabilityPolicy
from geometry_msgs.msg      import Quaternion
from geometry_msgs.msg      import QuaternionStamped
from std_msgs.msg           import Float64MultiArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from me133a_final.utils.TransformHelpers import *


#
#  GUI Slider Class
#
class SingleVariable(QWidget):
    def __init__(self, name, val, minval, maxval, callback):
        super().__init__()
        self.value    = val
        self.offset   = (maxval + minval) / 2.0
        self.slope    = (maxval - minval) / 200.0
        self.callback = callback
        self.initUI(name)

    def initUI(self, name):
        # Top-Left: Name
        label = QLabel(name)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setMinimumWidth(40)

        # Top-Right: Number
        self.number = QLabel("%6.3f" % self.value)
        self.number.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.number.setMinimumWidth(100)
        self.number.setStyleSheet("border: 1px solid black;")

        # Bottom: Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-100, 100)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setPageStep(5)
        slider.valueChanged.connect(self.valueHandler)
        slider.setValue(int((self.value - self.offset)/self.slope))

        # Create the Layout
        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addSpacing(10)
        hbox.addWidget(self.number)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(slider)

        self.setLayout(vbox)

    def valueHandler(self, value):
        self.value = self.offset + self.slope * float(value)
        self.number.setText("%6.3f" % self.value)
        self.callback(self.value)

    
class PanTiltGUI(QWidget):
    def __init__(self, angles0, callback):
        super().__init__()
        self.value    = angles0
        self.callback = callback
        [self.pan, self.tilt, self.dist] = angles0
        self.initUI(self.value)

    def initUI(self, angles):
        # Create the Pan/tilt variables
        vbox = QVBoxLayout()
        vbox.addWidget(SingleVariable('Pan', angles[0], -np.pi/4, np.pi/4, self.panHandler))
        vbox.addWidget(SingleVariable('Tilt', angles[1], -np.pi/4, np.pi/4, self.tiltHandler))
        vbox.addWidget(SingleVariable('Distance', angles[2], 0.0, 1.3, self.distHandler))

        self.setLayout(vbox)
        self.setWindowTitle('Pan/Tilt')
        self.show()

    def panHandler(self, value):
        self.pan = value
        self.value = [self.pan, self.tilt, self.dist]
        self.callback(self.value)

    def tiltHandler(self, value):
        self.tilt = value
        self.value = [self.pan, self.tilt, self.dist]
        self.callback(self.value)
        
    def distHandler(self, value):
        self.dist = value
        self.value = [self.pan, self.tilt, self.dist]
        self.callback(self.value)

    def kill(self, signum, frame):
        self.close()


#
#   GUI Node Class
#
class GUINode(Node):
    # Initialization.
    def __init__(self, name, initvalue, rate):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Prepare the publisher (latching for new subscribers).
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.pub_mark  = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)
        self.pub_angles = self.create_publisher(
            Float64MultiArray, '/bar_angles', quality)

        # Save the publisher rate and status.
        self.publishrate = rate
        self.publishing  = False

        # Create the FloatArray containing the pan/tilt angles of the bar.
        self.bar_angles = Float64MultiArray()
        self.setvalue(initvalue)
        self.pan = self.bar_angles.data[0]
        self.tilt = self.bar_angles.data[1]
        self.dist = self.bar_angles.data[2]

        # Save the bar orientation
        self.q = quat_from_R(Rotx(np.pi/2)@Rotx(self.tilt)@Roty(self.pan))

        # Create the bar marker.
        self.marker = Marker()
        self.marker.header.frame_id    = "world"
        self.marker.header.stamp       = self.get_clock().now().to_msg()

        self.marker.action             = Marker.ADD
        self.marker.ns                 = "bar"
        self.marker.id                 = 1
        self.marker.type               = Marker.CYLINDER
        self.marker.pose.orientation.x = self.q[1]
        self.marker.pose.orientation.y = self.q[2]
        self.marker.pose.orientation.z = self.q[3]
        self.marker.pose.orientation.w = self.q[0]
        self.marker.pose.position      = Point_from_p(pxyz(0.0,0.0,1.96)) # fixme to be the correct location
        self.marker.scale.x            = 0.05 # fixme
        self.marker.scale.y            = 0.05 # fixme
        self.marker.scale.z            = 1.5 # fixme
        self.marker.color.r            = 1.5
        self.marker.color.g            = 0.0
        self.marker.color.b            = 0.0
        self.marker.color.a            = 0.8     # Make transparent!

        # Create the marker array message.
        self.mark = MarkerArray()
        self.mark.markers.append(self.marker)


    # Run
    def run(self):
        # Prepare Qt.
        app = QApplication(sys.argv)
        app.setApplicationDisplayName("Bar Publisher")

        # Include a Qt Timer, setup up so every 500ms the python
        # interpreter runs (doing nothing).  This enables the ctrl-c
        # handler to be processed and close the window.
        timer = QTimer()
        timer.start(500)
        timer.timeout.connect(lambda: None)

        # Then setup the GUI window.  And declare the ctrl-c handler to
        # close that window.
        gui = PanTiltGUI(self.getvalue(), self.setvalue)
        signal.signal(signal.SIGINT, gui.kill)

        # Start the publisher in a separate thread.
        self.publishing = True
        thread = threading.Thread(target=self.publisher)
        thread.start()

        # Start the GUI window.
        self.get_logger().info("GUI starting...")
        status = app.exec_()
        self.get_logger().info("GUI ended.")

        # End the publisher.
        self.publishing = False
        thread.join()


    # Publisher Loop
    def publisher(self):
        # Create a timer to control the publishing.
        rate  = self.publishrate
        timer = self.create_timer(1/float(rate), self.publish)
        dt    = timer.timer_period_ns * 1e-9
        self.get_logger().info("Publishing with dt of %f seconds (%fHz)" %
                               (dt, rate))

        # Publish until told to stop.
        while self.publishing:
            rclpy.spin_once(self)

        # Destroy the timer and report.
        timer.destroy()        
        self.get_logger().info("Stopped publishing")


    # Publish the current value.
    def publish(self):
        # Grab the current time.
        now = self.get_clock().now()
        # Publish.
        self.marker.header.stamp = now.to_msg()

        # Recreate the bar marker.
        self.marker = Marker()
        self.marker.header.frame_id    = "world"
        self.marker.header.stamp       = self.get_clock().now().to_msg()

        self.marker.action             = Marker.ADD
        self.marker.ns                 = "bar"
        self.marker.id                 = 1
        self.marker.type               = Marker.CYLINDER
        self.marker.pose.orientation.x = self.q[1]
        self.marker.pose.orientation.y = self.q[2]
        self.marker.pose.orientation.z = self.q[3]
        self.marker.pose.orientation.w = self.q[0]
        self.marker.pose.position      = Point_from_p(pxyz(0.0,0.0,1.96)) # fixme to be the correct location
        self.marker.scale.x            = 0.05 # fixme
        self.marker.scale.y            = 0.05 # fixme
        self.marker.scale.z            = 1.5  # fixme
        self.marker.color.r            = 1.5
        self.marker.color.g            = 0.0
        self.marker.color.b            = 0.0
        self.marker.color.a            = 0.8     # Make transparent!

        # Create the marker array message.
        self.mark = MarkerArray()
        self.mark.markers.append(self.marker)

        self.pub_mark.publish(self.mark)
        self.pub_angles.publish(self.bar_angles)


    # Get/Set the value.
    def getvalue(self):
        # Get the value.
        return self.bar_angles.data
    def setvalue(self, value):
        # Set the value.
        self.bar_angles.data = value
        self.pan = self.bar_angles.data[0]
        self.tilt = self.bar_angles.data[1]
        self.dist = self.bar_angles.data[2]
        self.q = quat_from_R(Rotx(np.pi/2)@Rotx(self.tilt)@Roty(self.pan))
