#!/usr/bin/env python3

import rospy
import ast
from std_msgs.msg import Float32,String,Bool,UInt16
from Controller_Config import ALL_SETTINGS
SETTINGS = ALL_SETTINGS.learning_node
FILTERS = ALL_SETTINGS.filters
TOPICS = ALL_SETTINGS.topics


class VirtualController(object):
	def __init__(self,dtau = None):
		self.state_names = ['pos','vel','acc','jerk','F','dFdt','dt','timestamp']
		self.state = {}
		for key in self.state_names: self.state[key] = 0
		self.done = True
		self.start_pos = None # read from controller
		self.goal_pos = None
		self.CONTROLLER_STATUS = 'starting'
		# ====================================================
		# Init ROS Environment ===============================
		"""
		self.topic_damping_action = 'RL/damping_cmd'
		self.topic_settings = 'RL/settings'
		self.topic_force_reset = 'RL/force_reset'
		self.topic_controller_state = 'controller/controller_state'
		"""
		# Import Topics
		for key in TOPICS.keys():
			self.__dict__[key] = TOPICS[key]
		
		
		# Node -----------------------------------------------
		# Not needed in virtual since loop rate is determiend by learning algorithm
		#self.fs = SETTINGS['loop_rate']					# expected controller sample freq
		#self.dt = 1/self.fs 							# expected controller timestep
		#rospy.init_node(self.node_handler, anonymous=True) 	# start ROS node
		#self.loop_rate = rospy.Rate(self.fs)				# init ROS node freq
		
		# Subscribers ----------------------------------------
		self.sub_controller_state = rospy.Subscriber(self.topic_controller_state, String, self.controller_state_callback)
		
		# Publishers -----------------------------------------	
		self.damping_publisher = rospy.Publisher(self.topic_damping_action, Float32, queue_size=1)
		self.settings_publisher = rospy.Publisher(self.topic_settings, String, queue_size=1)
		self.reset_publisher = rospy.Publisher(self.topic_force_reset, Bool, queue_size=1)
	#######################################################
	# RL ENVIORMENT INTERFACE #############################
	#######################################################
	def get_state(self):
		v = self.state['vel']
		dvdt = self.state['acc'] 
		F = self.state['F']
		dFdt =self.state['dFdt'] 
		J = self.state['jerk']
		return v,dvdt,F,dFdt,J 
	
	def set_damping(self,Kd):
		#Kd_filt = FILTERS['damping'].sample(Kd)
		#self.damping_publisher.publish(Kd_filt)
		self.damping_publisher.publish(Kd)
	
	def load_settings(self,goal,max_dur,tol):
		msg_dict = {}
		msg_dict['posf'] = self.goal_pos # goal position
		msg_dict['velf'] = 0.0 # goal velocity
		msg_dict['accf'] = 0.0 # goal acceleration
		msg_dict['Ff'] = 0.0 # human satisfied with position
		msg_dict['dFdtf'] = 0.0 # human satisfied with position
		msg_dict['tol'] = tol # maximum error considered @ goal
		msg_dict['max_duration'] = max_dur # episode in seconds
		msg = str(msg_dict)
		self.settings_publisher.publish(msg)
		
	def report(self):
		# Report --------------------------------------------------------------
		try:
			print(f'\nCalculate_admittance [{self.CONTROLLER_STATUS}] [N={len(self.hist["F"])}] [{round(self.state["timestamp"],2)}]')
			print(f'\t| pos  : {round(self.start_pos,2)} => {round(self.state["pos"],2)} => {self.goal_pos}')#\t {np.round(np.max(self.hist["pos"]),2)}')
			print(f'\t| v    : {round(self.state["vel"],2)} ')#\t {np.round(np.max(self.hist["vel"]),2)}')
			print(f'\t| dvdt : {round(self.state["acc"],2)} ')#\t {np.round(np.max(self.hist["acc"]),2)}')
			print(f'\t| F    : {round(self.state["F"],2)} ')#\t {np.round(np.max(self.hist["F"]),2)}')
			print(f'\t| dFdt : {round(self.state["dFdt"],5)} ')#\t {np.round(np.max(self.hist["dFdt"]),5)}')
			print(f'\t| J    : {round(self.state["jerk"],5)}')# \t {np.round(np.max(self.hist["jerk"]),5)}')
			print(f'\t| dt   : {round(self.state["dt"],5)} ')#\t {np.round(np.max(self.hist["dt"]),5)}')	
			print(f'\t| done : {self.done} ')#\t {np.round(np.max(self.hist["dt"]),5)}')	
			print(f'\t| Kd   : {self.D[0,0]} ')#\t {np.round(np.max(self.hist["dt"]),5)}')	
		except Exception as e:
			print(f'report_error: {e}')
	#######################################################
	# CALLBACKS ###########################################
	#######################################################
	def controller_state_callback(self,msg):
		
		msg_str = msg.data
		rec_dict = ast.literal_eval(msg_str)
		for key in self.state_names:
			self.state[key] = rec_dict[key]
		self.done = rec_dict['done']
		self.start_pos = rec_dict['start_pos']
		self.CONTROLLER_STATUS = rec_dict['status']
		#print(f"{self.state['timestamp']} updating_state ")
	#"""
	def reset(self):
		print('\t|Reseting robot to start state...')
		msg = Bool()
		msg.data = False
		i = 0
		#self.reset_publisher.publish(msg)
		while True: 
			msg.data = True
			i +=1
			print(f'\r\t|[{i}] writing reset = {msg} [{self.CONTROLLER_STATUS},{self.done}]...',end='')
			self.reset_publisher.publish(msg)
			rospy.sleep(1)
			#self.loop_rate.sleep()
			if rospy.is_shutdown(): break
			if not self.done: break
			
		
		msg.data = False
		print(f'\n\t| !!! Robot reset {msg} !!!')
		self.reset_publisher.publish(msg)
	#"""
	
