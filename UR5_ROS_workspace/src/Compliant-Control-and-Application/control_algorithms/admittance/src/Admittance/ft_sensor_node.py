#!/usr/bin/env python3
# source ~/ws_test/devel/setup.bash && cd ~/ws_test/ && rosrun admittance ft_sensor_node.py
# rosrun admittance ft_sensor_node.py
"""
Author: Mason Smith
Date: 11/17/22
Description:
	Implementation of ROS publisher to read the wrench data from the RobotIQ FT300 sensor.
	Data from the FT300 is configured to stream over TCP on the UR5's IP address automatically on boot.
	See section "6.1.3. Data Stream" in FT300 instruction manual for more information.
	FT300 sends messages at 100hz to the stream
FT300 Manual: 
	https://robotiq.com/support/ft-300-force-torque-sensor
	[Universal Robots >> Documents >> FT 300-S Instruction Manual]
"""

def main():
	import rospy
	import socket
	import numpy as np
	from geometry_msgs.msg import WrenchStamped 
	
	#"""
	from Controller_Config import ALL_SETTINGS
	SETTINGS = ALL_SETTINGS.ft_node
	
	# Init node and publisher (IMPORT SETTINGS)
	pub = rospy.Publisher(SETTINGS['publish_topic'], WrenchStamped, queue_size=SETTINGS['que_sz'])
	rospy.init_node(SETTINGS['node_handler'], anonymous=True)
	loop_rate = rospy.Rate(SETTINGS['loop_rate']) 
	HOST = SETTINGS['host']#raw_input("Enter IP address of UR controller:") # The remote host
	PORT = SETTINGS['port'] # The same port as used by the server
	publish_zeros_on_error = SETTINGS['publish_zeros_on_error']
	#"""
	
	"""
	# Init node and publisher (CUSTOM SETTINGS)
	pub = rospy.Publisher('ft300_force_torque', WrenchStamped, queue_size=10) # topic name
	rospy.init_node('torque_force_sensor_data', anonymous=True)
	loop_rate = rospy.Rate(200) 	# publisher frequency
	HOST = '192.168.0.100'	# IP of the UR5
	PORT = 63351 			# The data stream port for the FT300 on UR5 ip 
	publish_zeros_on_error = False # handling rec timeout
	"""
	# Begin socket connection
	try:
		rospy.loginfo("Connecting to " + HOST)
		
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.connect((HOST, PORT))
			s.settimeout(0.5)
			rospy.loginfo("Connection established")
		
			while not rospy.is_shutdown():

				try:
					# probably better ways of recieving data
					# this works for now.
					data = str(s.recv(1024))
					if rospy.is_shutdown(): break
					if data:
						data = data.split(')')[-2]
						data = data.replace("(","")
						data = data.replace(" ","")
						data = data.replace("b'","")
						data = data.split(",")
						data = np.array([float(val) for val in data]) 
						data = data 
						
						msg = WrenchStamped()
						msg.wrench.force.x = data[0]
						msg.wrench.force.y = data[1]
						msg.wrench.force.z = data[2]
						msg.wrench.torque.x = data[3]
						msg.wrench.torque.y = data[4]
						msg.wrench.torque.x = data[5]
						pub.publish(msg)
						rospy.loginfo(data)
					else:
						rospy.loginfo(f'empty: {data}')
						
					
					
				except Exception as e:
					rospy.loginfo(f'\n\nread error \n\t| {e} \n\n')
					if publish_zeros_on_error:
						data = np.zeros(6)
						msg = WrenchStamped()
						msg.wrench.force.x = data[0]
						msg.wrench.force.y = data[1]
						msg.wrench.force.z = data[2]
						msg.wrench.torque.x = data[3]
						msg.wrench.torque.y = data[4]
						msg.wrench.torque.x = data[5]
						pub.publish(msg)
					
				
				loop_rate.sleep()
		rospy.loginfo('closed socket...')
	except Exception as e:
		rospy.loginfo(f"No connection \n\t| {e}")
		
if __name__ =="__main__":
	main()


