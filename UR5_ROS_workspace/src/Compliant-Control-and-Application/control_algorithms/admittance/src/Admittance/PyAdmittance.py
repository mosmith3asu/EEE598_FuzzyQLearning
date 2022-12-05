#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Pose,Twist, Wrench, WrenchStamped
from cartesian_state_msgs.msg import PoseTwist
from tf import TransformListener, listener
from std_msgs.msg import String,Float32, Bool
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler,quaternion_multiply,quaternion_matrix,euler_matrix
from geometry_msgs.msg import Quaternion
import time 
import matplotlib.pyplot as plt
#from filters import RT_lowpass_butter
from scipy.signal import butter, lfilter, freqz
from scipy import signal
import json
import copy

from admittance.Controller_Config import ALL_SETTINGS
SETTINGS = ALL_SETTINGS.admittance_node
FILTERS = ALL_SETTINGS.filters
TOPICS = ALL_SETTINGS.topics
PUBLIC = ALL_SETTINGS.public


class Admittance(object):
	def __init__(self,dtau = None):
		self.CONTROLLER_STATUS = 'stopped'
		
		# ====================================================
		# IMPORT TOPIC DEFINITIONs	========================
		print('Initializing Admittance Object...')
		self.node_handler = SETTINGS['node_handler']
		
		for key in TOPICS.keys():
			self.__dict__[key] = TOPICS[key]
		
		"""
		self.base_link = "/base_link"
		self.end_link = "/wrist_3_link"
		self.node_handler = 'admittance_controller'
		self.topic_arm_command = '/cartesian_velocity_controller/command_cart_vel'
		self.topic_arm_state = '/cartesian_velocity_controller/ee_state'
		self.topic_wrench_state = '/ft300_force_torque' # '/wrench'
		self.topic_wrench_state = '/ft300_force_torque' # '/wrench'
		self.topic_controller_state = 'controller/controller_state'
		self.topic_damping_action = 'RL/damping_cmd'
		self.topic_settings = 'RL/settings'
		self.topic_force_reset = 'RL/force_reset'
		"""
		
		#self.self.topic_RL_state = ''
		# ====================================================
		# INITIATE STATE MEMORY ==============================
		self.state_names = ['pos','vel','acc','jerk','F','dFdt','dt','timestamp']
		self.state = {}				# current named state of the system
		self.hist = {}					# list for each state at timestep during this episode
		self.terminal = {}				# conditions for terminating robot controller (@ goal)
		self.start_time = time.time()		# time controller object initiated
		self.last_sample_time = time.time() # time reference for dynamically computing timestep (dt)
		self.done = True				# status of controller (needs reset if True)
		self.start_pos = None
		self.goal_pos = None
		self.hist_sz = 1e6
		# ====================================================
		# INITIATE DESIRED POSE ==============================
		val_ignore = 0.0001 					# does not matter since it is along axis of travel
		point_down_quat = [0.0, 1.0, 0.0, 0.0] 		# rotation of ee
		desired_position = [val_ignore,0.3,0.5] 	# constrained positions along y,z
		desired_orientation_quat = point_down_quat 	# constrained rotation
		self.desired_pose = desired_position + list(desired_orientation_quat )
		
		# ====================================================
		# Define dynamics ====================================
		M_FTsensor,M_gripper = 0.3, 0.925  							  # mass of gripper and FT sensor
		M_ee = M_gripper + M_FTsensor									  # cumulative mass of ee
		M_const, D_const, K_const  = 5, 50, 100 						  # constrained impedance values
		D0 = SETTINGS['D_default']
		self.M = np.diag([M_ee,		M_const,M_const,M_const,M_const,M_const]) # Mat6x6d interial matrix
		self.D = np.diag([D0,		D_const,D_const,D_const,D_const,D_const]) # Mat6x6d damping matrix
		self.K = np.diag([val_ignore, K_const,K_const,K_const,K_const,K_const]) # Mat6x6d stiffness matrix
		
		# ====================================================
		# INITIATE AUXILLARY VARIABLES =======================
		self.arm_desired_twist_adm_ = np.zeros(6) 	# Vector6D of velocities sent to robot velocity controller
		self.current_position = np.zeros(3)		# Vector3D current cartesian position of ee
		self.current_orientation_quat = np.zeros(4)	# Vector4D current quat orientation of ee
		self.current_ext_wrench_vec = np.zeros(6)	# Vector6D human's interaction force on ee
		self.error = np.zeros(6) 				# Vector6D error between desired and current position
		
		# ====================================================
		# Init ROS Environment ===============================
		
		# Node -----------------------------------------------
		self.fs = SETTINGS['loop_rate']					# expected controller sample freq
		self.dt = 1/self.fs 							# expected controller timestep
		rospy.init_node(self.node_handler, anonymous=True) 	# start ROS node
		self.loop_rate = rospy.Rate(self.fs)				# init ROS node freq
		
		# Subscribers ----------------------------------------
		self.sub_arm_state_ = rospy.Subscriber(self.topic_arm_state, PoseTwist, self.state_arm_callback)
		self.sub_wrench_state_ = rospy.Subscriber(self.topic_wrench_state, WrenchStamped, self.state_wrench_callback)
		self.sub_damping_action = rospy.Subscriber(self.topic_damping_action, Float32, self.damping_action_callback)
		self.sub_topic_settings = rospy.Subscriber(self.topic_settings, String, self.settings_callback)
		self.sub_topic_force_reset = rospy.Subscriber(self.topic_force_reset, Bool, self.reset_callback)
	
	
		# Publishers -----------------------------------------	
		self.pub_arm_cmd_ = rospy.Publisher(self.topic_arm_command, Twist, queue_size=5)
		self.pub_controller_state = rospy.Publisher(self.topic_controller_state,String, queue_size=5)
		
		

		# ====================================================
		# FILTERS ============================================
		""" GET SAMPLING FREQUENCIES FROM TOPIC USING:  'rostopic bw /<name>'  """
		fs = 1/self.dt
		self.filters = {}
		self.filters['acc'] 	= FILTERS['acc']	# RT_lowpass_butter(fs=fs,fc=fs/35.,order=5,enable=True)
		self.filters['F'] 		= FILTERS['F'] 	#RT_lowpass_butter(fs=fs,fc=fs/80.,order=2,enable=True) 
		self.filters['jerk'] 	= FILTERS['jerk']	#RT_lowpass_butter(fs=fs,fc=fs/100.,order=2,enable=True)
		self.filters['dFdt'] 	= FILTERS['dFdt']	#RT_lowpass_butter(fs=fs,fc=fs/100.,order=2,enable=True)
		
		
		
		# ====================================================
		# Guards =============================================
		
		self.t_arm_ready_ = False 		# check external is functioning
		self.base_world_ready_ = False 	# check external is functioning
		self.world_arm_ready_ = False		# check external is functioning
		
		self.ws_limits = np.array([[-0.5,0.2,0.4], [0.5,0.4,0.6]]) # position limits (attract to middle)
		self.arm_max_damping_ = SETTINGS['D_max'] 	# maximum R action to change damping values
		self.arm_max_vel_ = SETTINGS['v_max']		# maximum speed arm can travel
		self.arm_max_acc_ = SETTINGS['a_max']		# maximum acc arm can achieve


		# ====================================================
		# READ FOR CONNECTION ================================
		self.wait_for_transformations() # wait for external control to connect
		#self.wait_for_ext_reset()
	
		
	#######################################################
	# ROS INTERFACE #######################################
	#######################################################
	"""
	def wait_for_ext_reset(self):
		idisp = 0
		buffer_disp = ['|','/','-','\\']
		while True:
			img = buffer_disp[idisp%4]
			print(f'\r\t| ({img}) waiting for external reset ({img}) ', end ='')
			idisp +=1
			if rospy.is_shutdown(): break
			#if self.CONTROLLER_STATUS == 'resetting': break
			if self.CONTROLLER_STATUS == 'ready': break
			rospy.sleep(1.0)
		print('\t| Reset successfull')
	"""
	def publish_state(self):
		msg_dict = copy.deepcopy(self.state)
		msg_dict['done']=self.done
		msg_dict['start_pos'] = self.start_pos
		msg_dict['status'] = self.CONTROLLER_STATUS
		msg = str(msg_dict)
		self.pub_controller_state.publish(msg)
		
	def wait_for_transformations(self):
		print('Getting transform')
		listener = TransformListener() 
		print('\t| Waiting for TF from: base_link to: wrist_3_link')
		while self.get_rotation_matrix(listener,self.base_link,self.end_link) is not None:
			rospy.sleep(0.5)
			if rospy.is_shutdown(): break
			
		print('\t| Waiting for arm state...')
		while np.all(self.current_position==0):
			rospy.sleep(0.5)
			if rospy.is_shutdown(): break
		
		self.t_arm_ready_ = True
		self.base_world_ready_ = True
		self.world_arm_ready_ = True 
		print('\t| The Force/Torque sensor is ready to use')
		#self.reset()
		
		self.start_time = time.time()
		self.state['pos'] = self.current_position[0]
		self.state['vel'] = self.arm_desired_twist_adm_[0]
		self.state['acc'] = 0
		self.state['jerk'] = 0
		self.state['F'] = self.current_ext_wrench_vec[0]
		self.state['dFdt'] = 0
		self.state['dt'] = self.dt
		self.state['timestamp'] = time.time() - self.start_time
		self.last_sample_time = time.time()
		self.done = False
		for key in self.state.keys(): self.hist[key] = [self.state[key]]
		
		self.start_pos = self.state['pos']
		self.CONTROLLER_STATUS = 'ready'
				
	def get_rotation_matrix(self,listener,from_frame,to_frame,check=False):
		try:
			t = listener.getLatestCommonTime(from_frame,to_frame) # "/base_link", "/map"
			position, quaternion = listener.lookupTransform(from_frame,to_frame, t)
			rot_mat = quaternion_matrix(quaternion)[:3,:3]
			if check: return True
			else: return rot_mat 
		except Exception as e:
			#print(f'\t| Waiting for transform {e}')
			return None
		
		
	def compute_admittance(self):
		def quaternion_error(q1,q2): 
			q1 = np.copy(q1)
			q2_inv = np.copy(q2)
			q2_inv[3] = - q2_inv[3] 
			qerr = quaternion_multiply(q1,q2_inv)
			return qerr/np.linalg.norm(qerr)
			
		def quaternion2euler(q): return euler_from_quaternion(q)
		
		# UNPACK VARS --------------------------------------------------- 
		current_orientation_quat = self.current_orientation_quat
		current_position = self.current_position
		current_orientation_quat = self.current_orientation_quat
		desired_orientation_quat = np.array(self.desired_pose[3:])
		desired_position = np.array(self.desired_pose[:3])
		arm_desired_twist_adm_ = self.arm_desired_twist_adm_
		wrench_external_ = self.current_ext_wrench_vec
		D_ = self.D
		M_ = self.M
		K_ = self.K
		
		# Check timestep ------------------------------------------------- 
		dt = time.time() - self.last_sample_time
		self.last_sample_time = time.time()
		#if dt < 0.5*self.dt or dt > 1.5*self.dt: 	
		#	return None # break loop if invalid dt
		#dt = max(dt,0.5*self.dt)
		#dt = min(dt,1.5*self.dt)
		
		# Translation error w.r.t. desired equilibrium ------------------- 
		self.error[0:3] = current_position -desired_position
		
		# Rotation error w.r.t. desired equilibrium ----------------------
		error_quat = quaternion_error(current_orientation_quat,desired_orientation_quat)
		# error_quat = error_quat/np.linalg.norm(error_quat)
		self.error[3:] = quaternion2euler(error_quat)
		
		# Convert to axis angles -----------------------------------------
		coupling_wrench_arm = np.dot(D_,arm_desired_twist_adm_.T) + np.dot(K_,self.error.T)
		wrench_net = ( - coupling_wrench_arm  + wrench_external_)
		arm_desired_accelaration = np.dot(np.linalg.inv(M_),wrench_net.T) 
		a_acc_norm = np.linalg.norm(arm_desired_accelaration[:3])
		if a_acc_norm > self.arm_max_acc_: arm_desired_accelaration[:3] *= (self.arm_max_acc_/a_acc_norm)
		
		# FILTER ACC ------------------------------------------------------ 
		arm_desired_accelaration[0] = self.filters['acc'].sample(arm_desired_accelaration[0])
		
		# Integrate for velocity based interface rospy.Rate --------------- 
		self.arm_desired_twist_adm_ += arm_desired_accelaration * dt # self.dt
		
		# Calculate derivatives ---------------------------------------------
		dFdt = (self.state['F'] - wrench_external_[0])/dt #self.dt
		jerk = (self.state['acc'] - arm_desired_accelaration[0])/dt #self.dt
		
		if isinstance(dFdt,np.ndarray): dFdt = dFdt[0]
		if isinstance(jerk,np.ndarray): jerk = jerk[0]
		
		# Update Current state for RL -------------------------------------- 
		self.state['dFdt'] = self.filters['dFdt'].sample(dFdt)[0]
		self.state['jerk'] = self.filters['jerk'].sample(jerk)[0]
		self.state['acc'] = arm_desired_accelaration[0]
		self.state['vel'] = arm_desired_twist_adm_[0]
		self.state['pos'] = current_position[0]
		self.state['F'] = wrench_external_[0]
		self.state['dt'] = dt # time.time() - self.last_sample_time 
		self.state['timestamp'] = time.time() - self.start_time
		
		#for key in self.max_state.keys(): self.max_state[key] = max(self.max_state[key],abs(self.state[key]))
		for key in self.hist.keys():
			self.hist[key].append(self.state[key])
			if len(self.hist[key])> self.hist_sz: self.hist[key] = self.hist[key][1000:]
	
		
		
		
		
	def send_command(self):
		scale = 1.0 		
		
		# Check velocity limits ------------------
		v_norm = np.linalg.norm(self.arm_desired_twist_adm_[:3])
		if v_norm > self.arm_max_vel_: self.arm_desired_twist_adm_[:3] *= (self.arm_max_vel_ / v_norm)
		v_norm = np.linalg.norm(self.arm_desired_twist_adm_[3:])
		if v_norm > self.arm_max_vel_: self.arm_desired_twist_adm_[3:] *= (self.arm_max_vel_ / v_norm)
		
		# Check ws limits ------------------
		"""
		for ixyz, val in enumerate(self.current_position):
			exceeds_LB = (val < self.ws_limits[0,ixyz])
			exceeds_UB = (val > self.ws_limits[1,ixyz])
			if exceeds_LB or exceeds_UB: self.K[0,0] = 100
			else: self.K[0,0] = 0.001 # negligable
		"""
			
		# Update value ------------------
		arm_twist_cmd = Twist()
		arm_twist_cmd.linear.x  = self.arm_desired_twist_adm_[0]*scale
		arm_twist_cmd.linear.y  = self.arm_desired_twist_adm_[1]*scale
		arm_twist_cmd.linear.z  = self.arm_desired_twist_adm_[2]*scale
		arm_twist_cmd.angular.x = self.arm_desired_twist_adm_[3]*scale
		arm_twist_cmd.angular.y = self.arm_desired_twist_adm_[4]*scale
		arm_twist_cmd.angular.z = self.arm_desired_twist_adm_[5]*scale
		self.pub_arm_cmd_.publish(arm_twist_cmd)
		return arm_twist_cmd
		
	#######################################################
	# CALLBACKS ###########################################
	#######################################################
	def reset_callback(self,msg):
		#if self.CONTROLLER_STATUS =='starting':
		#	self.CONTROLLER_STATUS =='ready'
		#else:
		res = msg.data
		print(f'\n reset_callback [{res}{self.done}]')
		if res:
			print('Recieved reset callback')
			rospy.sleep(1)
			self.done = True
			#self.CONTROLLER_STATUS = 'resetting'
			#self.reset()
		elif res and not self.done:
			print('ERROR: CALLED RESSET BEFORE DONE')
		
	def settings_callback(self,msg):
		#if self.done: return None
		msg_str = msg.data
		rec_dict =  ast.literal_eval(msg_str)
		self.goal_pos = rec_dict['posf']  # goal position
		terminal['pos'] = rec_dict['posf']  # goal position 
		terminal['vel'] = rec_dict['velf'] = 0.0 # goal velocity
		terminal['acc'] = rec_dict['accf'] = 0.0 # goal acceleration
		terminal['F'] = rec_dict['Ff'] = 0.0 # human satisfied with position
		terminal['dFdt'] = rec_dict['dFdtf'] = 0.0 # human satisfied with position
		terminal['tol'] = rec_dict['tol'] = 0.1 # maximum error considered @ goal
		terminal['max_duration'] = rec_dict['max_duration'] = max_epi_duration # seconds
		self.set_terminal(terminal)

	def damping_action_callback(self,msg):
		if self.done: return None
		val = msg.data
		self.set_damping(val)
	def state_arm_callback(self,msg):
		self.current_position = np.array([
			msg.pose.position.x,
			msg.pose.position.y,
			msg.pose.position.z
		])
		self.current_orientation_quat = np.array([
			msg.pose.orientation.x,
			msg.pose.orientation.y,
			msg.pose.orientation.z,
			msg.pose.orientation.w
		])
		
		
	def state_wrench_callback(self,msg): 
		if self.done: return None
		# Settings --------------------------
		torque_deadzone_lower_limit = 8.0
		force_deadzone_lb = SETTINGS['F_lb']
		force_deadzone_ub = SETTINGS['F_ub']
		force_scale = 1.0 

		# Read raw value
		wrench_new = np.zeros(6)
		wrench_new[0] = - msg.wrench.force.x
		#wrench_new[1] = msg.wrench.force.y
		#wrench_new[2] = msg.wrench.force.z
		#wrench_new[3] = msg.wrench.torque.x
		#wrench_new[4] = msg.wrench.torque.y
		#wrench_new[5] = msg.wrench.torque.z
				
		# Filter measurement -----------------------------------------
		wrench_new[0] = self.filters['F'].sample(wrench_new[0])
	
		# Check bounds and errors ---------------------------------------
		for i in range(3):
			# apply offset
			
			
			# lower force bound
			if abs(wrench_new[i]) < force_deadzone_lb: wrench_new[i] = 0
			elif wrench_new[i] >0: wrench_new[i] -= force_deadzone_lb
			else: wrench_new[i] += force_deadzone_lb
			
			# upper force bound
			if abs(wrench_new[i]) > force_deadzone_ub: 
				if wrench_new[i] >0: wrench_new[i] = force_deadzone_ub - force_deadzone_lb
				else: wrench_new[i] = force_deadzone_ub + force_deadzone_lb
			
			# lower torque bound
			if abs(wrench_new[i+3]) < torque_deadzone_lower_limit: wrench_new[i+3] = 0
			elif wrench_new[i+3] >0: wrench_new[i+3] -= torque_deadzone_lower_limit
			else: wrench_new[i+3] += torque_deadzone_lower_limit
		wrench_new = np.nan_to_num(wrench_new)
			
		# Update wrench
		self.current_ext_wrench_vec = wrench_new
	#######################################################
	# RL ENVIORMENT INTERFACE #############################
	#######################################################
	def reset(self):

		print(f'Resetting to pos=>{self.start_pos}')
		
		# MOVE TO START ----------------------------------------------------
		self.CONTROLLER_STATUS = 'resetting'
		self.current_ext_wrench_vec = np.zeros(6)
		self.K[0,0] = 25.
		self.D[0,0] = 5.
		timeout = 30 # seconds
		iz,zoff = 2, 0.1
		
		# Move slightly up for signal
		self.K[0,0] = 50.
		self.desired_pose[2] += zoff
		self.desired_pose[0] = self.current_position[0] 
		tstart = time.time()
		while time.time()-tstart < timeout:
			self.compute_admittance()
			self.send_command() 
			self.publish_state()
			self.loop_rate.sleep()
			self.report()
			if rospy.is_shutdown(): break
			if abs(self.current_position[iz] - self.desired_pose[iz]) <0.01: break
		
		# Move to start X 
		self.K[0,0] = 25.
		self.desired_pose[0] = self.start_pos
		tstart = time.time()
		while time.time()-tstart < timeout:
			self.compute_admittance()
			self.send_command() 
			self.publish_state()
			self.loop_rate.sleep()
			self.report()
			if rospy.is_shutdown(): break
			if abs(self.state['pos']- self.start_pos) <0.01: break
		
		
		# Move slightly down for signal
		self.K[0,0] = 50.
		self.desired_pose[2] += -zoff
		self.desired_pose[0] = self.current_position[0] 
		tstart = time.time()
		while time.time()-tstart < timeout:
			self.compute_admittance()
			self.send_command() 
			self.publish_state()
			self.loop_rate.sleep()
			self.report()
			if rospy.is_shutdown(): break
			if abs(self.current_position[iz] - self.desired_pose[iz]) <0.01: break
		
		
		# RESET VARS ----------------------------------------------------
		self.K[0,0] = 0.0001 # set back to negligable stiffness
		self.desired_pose[0] = 0.0 # set back to default desired pos
		self.start_time = time.time()
		self.state['pos'] = self.start_pos
		self.state['vel'] = 0
		self.state['acc'] = 0
		self.state['jerk'] = 0
		self.state['F'] = 0
		self.state['dFdt'] = 0
		self.state['dt'] = self.dt
		self.state['timestamp'] = time.time() - self.start_time
		self.last_sample_time = time.time()
		for key in self.state.keys(): self.hist[key] = [self.state[key]]
		self.arm_desired_twist_adm_ = np.zeros(6)
		self.done = False
		self.CONTROLLER_STATUS = 'ready'
		self.publish_state()
		
	def set_terminal(self,terminal_dict):
		self.goal_pos = terminal_dict['pos']
		self.terminal['pos'] = terminal_dict['pos']
		self.terminal['vel'] = terminal_dict['vel']
		self.terminal['acc'] = terminal_dict['acc']
		self.terminal['F'] = terminal_dict['F']
		self.terminal['dFdt'] = terminal_dict['dFdt']
		self.terminal['tol'] = terminal_dict['tol']
		self.terminal['max_duration'] = terminal_dict['max_duration']
			
	def check_terminal(self): # Check at terminal state
		checks = ['vel','acc','F','dFdt']
		tol = self.terminal['tol']
		
		#is_in_start_state = (abs(self.state['pos']-self.terminal['pos']) < 0.1)
		is_past_goal = (self.state['pos']>self.terminal['pos'])
		within_var_tol = [abs(self.state[key]-self.terminal[key]) < tol for key in checks]
		is_terminal_state = np.all(within_var_tol)
		exceeds_duration = (self.state['timestamp'] > self.terminal['max_duration'])
		
		if is_past_goal:
			#print('\n\nPAST GOAL\n\n')
			if is_terminal_state or exceeds_duration:
				print('\n\n########TERMINAL############\n\n')
				self.done = True
				self.CONTROLLER_STATUS = 'idle'
				#self.stop_all()
		return self.done
		
	def get_state(self):
		v = self.state['vel']
		dvdt = self.state['acc'] 
		F = self.state['F']
		dFdt =self.state['dFdt'] 
		J = self.state['jerk']
		return v,dvdt,F,dFdt,J 
		
	def set_damping(self,D):
		self.D[0,0] = min(D,self.arm_max_damping_)
		return D
		
	def step(self):
		if not rospy.is_shutdown():
			print('Stepping Controller...')
			self.compute_admittance()
			cmd = self.send_command() 
			#self.loop_rate.sleep()
			
	#######################################################
	# CONTROLLER HANDLERS #################################
	#######################################################
	def report(self):
		# Report --------------------------------------------------------------
		print(f'\nCalculate_admittance [{self.CONTROLLER_STATUS}] [N={len(self.hist["F"])}]')
		print(f'\t| pos  : {round(self.state["pos"],2)} => {self.goal_pos}')#\t {np.round(np.max(self.hist["pos"]),2)}')
		print(f'\t| v    : {round(self.state["vel"],2)} ')#\t {np.round(np.max(self.hist["vel"]),2)}')
		print(f'\t| dvdt : {round(self.state["acc"],2)} ')#\t {np.round(np.max(self.hist["acc"]),2)}')
		print(f'\t| F    : {round(self.state["F"],2)} ')#\t {np.round(np.max(self.hist["F"]),2)}')
		print(f'\t| dFdt : {round(self.state["dFdt"],5)} ')#\t {np.round(np.max(self.hist["dFdt"]),5)}')
		print(f'\t| J    : {round(self.state["jerk"],5)}')# \t {np.round(np.max(self.hist["jerk"]),5)}')
		print(f'\t| dt   : {round(self.state["dt"],5)} ')#\t {np.round(np.max(self.hist["dt"]),5)}')	
		print(f'\t| done : {self.done} ')#\t {np.round(np.max(self.hist["dt"]),5)}')	
		print(f'\t| Kd   : {self.D[0,0]} ')#\t {np.round(np.max(self.hist["dt"]),5)}')	
	
	def stop_all(self):	
		arm_twist_cmd = Twist()	
		arm_twist_cmd.linear.x  = 0
		arm_twist_cmd.linear.y  = 0
		arm_twist_cmd.linear.z  = 0
		arm_twist_cmd.angular.x = 0
		arm_twist_cmd.angular.y = 0
		arm_twist_cmd.angular.z = 0
		self.pub_arm_cmd_.publish(arm_twist_cmd)


	def run(self):
		print('Running Controller...')
		while not rospy.is_shutdown():
			self.check_terminal()
			if not self.done:
				self.compute_admittance()
				cmd = self.send_command() 
			else:
				self.reset()	
			
			self.report()
			
			# write to topic ------------------------------------------------------
			self.publish_state()
			self.loop_rate.sleep()
		#self.stop_all()
		#self.loop_rate.sleep()
		
		nkeys = len(list(self.hist.keys()))
		fig,axs = plt.subplots(nkeys,1)
		for i, key in enumerate(self.hist.keys()):
			axs[i].plot(self.hist[key])
			"""
			if key in ['jerk','dFdt','F','acc']:
				#yfilt = self.filters[key].test(self.hist[key])
				self.filters[key].enable = True
				yfilt = [self.filters[key].sample(self.hist[key][i]) for i in range(len(self.hist[key]))]
				axs[i].plot(yfilt)
			"""
			axs[i].set_ylabel(key)
		plt.show()
			
	
			
if __name__=="__main__":
	terminal = {}
	terminal['pos'] = PUBLIC['pos_des'] # goal position
	terminal['vel'] = 0.0 # goal velocity
	terminal['acc'] = 0.0 # goal acceleration
	terminal['F'] = 0.0 # human satisfied with position
	terminal['dFdt'] = 0.0 # human satisfied with position
	terminal['tol'] = PUBLIC['tol_des'] # maximum error considered @ goal
	terminal['max_duration'] = 10 # seconds
		
	controller = Admittance() 
	controller.set_terminal(terminal)
	controller.run()
		
def bound(val,bnds,test=False):
	if test:return (val>=bds[0] and val <= bnds[1])
	return min(max(val,bnds[0]),bnds[1])

		
