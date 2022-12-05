#!/usr/bin/env python3
# source ~/ws_test/devel/setup.bash && cd ~/ws_test/ && rosrun admittance ft_sensor_node.py
# rosrun admittance ft_sensor_node.py
import rospy
import socket
#from std_msgs.msg import Float32
from time import gmtime, strftime
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import WrenchStamped 
import time
import numpy as np

# Init node and publisher
#pub = rospy.Publisher('ft300_force_torque', Wrench, queue_size=10)
pub = rospy.Publisher('ft300_force_torque', WrenchStamped, queue_size=10)
rospy.init_node('torque_force_sensor_data', anonymous=True)
rate = rospy.Rate(110) 

bias = np.array([3,0,2.1,0,0,0])


HOST = '192.168.0.100'#raw_input("Enter IP address of UR controller:") # The remote host
PORT = 63351 # The same port as used by the server
#PORT = 63350 # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
	last_publish = time.time()
	print("Connecting to " + HOST)
	
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.connect((HOST, PORT))
		s.settimeout(1.0)
		print("Publishing")
		while not rospy.is_shutdown():
			try:
				data = str(s.recv(1024))
				if data:
					data = data.split(')')[-2]
					data = data.replace("(","")
					data = data.replace(" ","")
					data = data.replace("b'","")
					data = data.split(",")
					data = np.array([float(val) for val in data]) 
					data = data - bias
					
						
					msg = WrenchStamped()
					msg.wrench.force.x = data[0]
					msg.wrench.force.y = data[1]
					msg.wrench.force.z = data[2]
					msg.wrench.torque.x = data[3]
					msg.wrench.torque.y = data[4]
					msg.wrench.torque.x = data[5]
					pub.publish(msg)
			
				rospy.loginfo(data)
				rate.sleep()
			except Exception as e:
				rospy.loginfo(f'\n\nread error \n\t| {e} \n\n')
		"""
		while not rospy.is_shutdown():
			data = str(s.recv(1024))
			data = data.split(')')[-2]
			data = data.replace("(","")
			data = data.replace(" ","")
			data = data.replace("b'","")
			data = data.split(",")
			data = [float(val) for val in data]
			
			msg = WrenchStamped()
			msg.wrench.force.x = data[0]
			msg.wrench.force.y = data[1]
			msg.wrench.force.z = data[2]
			msg.wrench.torque.x = data[3]
			msg.wrench.torque.y = data[4]
			msg.wrench.torque.x = data[5]
			pub.publish(msg)
			#data = data.replace("(","")
			#data = data.replace(")","\n")
			rospy.loginfo(data)
			rate.sleep()
			#f.write(data)
		"""
	print('closed socket...')
except Exception as e:
	print (f"No connection \n\t| {e}")
