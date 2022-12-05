#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Pose,Twist, Wrench
from geometry_msgs.msg import WrenchStamped 
from cartesian_state_msgs.msg import PoseTwist
from tf import TransformListener, listener
from std_msgs.msg import String
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler,quaternion_multiply,quaternion_matrix,euler_matrix
from geometry_msgs.msg import Quaternion
import time 
import matplotlib.pyplot as plt
from filters import RT_lowpass_butter



class Admittance(object):
	def __init__(self,dtau = None):
		
		
		
		# PARAMS		====================
		print('Initializing Admittance Object...')
		self.base_link = "/base_link"
		self.end_link = "/wrist_3_link"
		self.topic_arm_command = '/cartesian_velocity_controller/command_cart_vel'
		self.topic_arm_state = '/cartesian_velocity_controller/ee_state'
		#self.topic_wrench_state = '/wrench'
		#self.topic_wrench_state = '/wrench_fake'
		self.topic_wrench_state = '/ft300_force_torque'
		self.node_handler = 'admittance_controller'
		

		self.last_sample_time = time.time() 
		self.x_start = - 0.4
		self.x_end = 0.3
		self.ws_limits = np.array([[-0.5,0.2,0.4], [0.5,0.4,0.6]])
		
		# ====================================================
		# INITIATE STATE VARIABLES ===========================
		self.state_names = ['pos','vel','acc','jerk','F','dFdt','dt']
		self.state = {}
		self.state['pos'] = 0
		self.state['vel'] = 0
		self.state['acc'] = 0
		self.state['jerk'] = 0
		self.state['F'] = 0
		self.state['dFdt'] = 0
		self.state['dt'] = 0
		
		self.max_state = {}
		self.max_state['pos'] = 0
		self.max_state['vel'] = 0
		self.max_state['acc'] = 0
		self.max_state['jerk'] = 0
		self.max_state['F'] = 0
		self.max_state['dFdt'] = 0
		self.max_state['dt'] = 0
		
		self.hist = {}
		self.hist['pos'] = []
		self.hist['vel'] = []
		self.hist['acc'] = []
		self.hist['jerk'] = []
		self.hist['F'] = []
		self.hist['dFdt'] = []
		self.hist['dt'] = []
		
		
		
		self.desired_position = [0.0,0.3,0.5]
		self.desired_orientation_quat = [0.0, 1.0, 0.0, 0.0]
		self.desired_pose = self.desired_position + list(self.desired_orientation_quat )
		self.workspace_limits: [-0.50, 0.50, 0.25, 0.80, 0.30, 0.75]
		
		# Define dynamics =====================
		M_FTsensor = 0.3
		M_gripper = 0.925 
		M_ee = M_gripper + M_FTsensor
		M_constrained = 5
		D_constrained = 50
		K_constrained = 100
		self.M = np.diag([M_ee,	M_constrained,M_constrained,M_constrained,M_constrained,M_constrained])
		self.D = np.diag([1.0,	D_constrained,D_constrained,D_constrained,D_constrained,D_constrained])
		self.K = np.diag([0.0001, K_constrained,K_constrained,K_constrained,K_constrained,K_constrained])
		
		self.D_limits = [0,20]

		# Initializing the class variables: ====================
		self.rot_matrix = np.zeros([6,6])
		self.arm_position_ = np.zeros(3) 			# Vector3D
		self.arm_orientation_ = np.zeros(4) 		# Quaternion
		self.arm_twist_ = np.zeros(6) 			# Vector6D
		self.wrench_external_= np.zeros(6)  		# Vector6D
		self.arm_desired_twist_adm_ = np.zeros(6) 	# Vector6D Twist()
		self.arm_desired_accelaration_ = np.zeros(6) # Vector6D
		self.current_position = np.zeros(3)
		self.current_orientation_quat = np.zeros(4)
		self.current_ext_wrench = Wrench()
		self.current_ext_wrench_vec = np.zeros(6)
		self.initial_ext_wrench_vec = np.zeros(6)
		
		# Init Node 	====================
		self.dt = 0.01 if dtau is None else dtau
		rospy.init_node(self.node_handler, anonymous=True)
		self.loop_rate = rospy.Rate(1/self.dt)
		
		# Subscribers 	====================
		self.sub_arm_state_ = rospy.Subscriber(self.topic_arm_state, PoseTwist, self.state_arm_callback)
		self.sub_wrench_state_ = rospy.Subscriber(self.topic_wrench_state, WrenchStamped, self.state_wrench_callback)
		
		# Listeners	====================
		self.listener_ft_ = TransformListener()
		self.listener_control_ = TransformListener()
		self.listener_arm_ = TransformListener()
		
		# Publisher	====================		
		self.pub_arm_cmd_ = rospy.Publisher(self.topic_arm_command, Twist, queue_size=5)
		
		self.desired_pose_position_= np.zeros(7) 	# Vector7D
		self.error = np.zeros(6) 				# Vector6D
	
		# TF: Transform from base_link to world ====================
		self.rotation_base_ = np.zeros([6,6])
		
		
		# FILTERS ========================
		fs = 1/self.dt
		self.filters = {}
		self.filters['F'] 		= RT_lowpass_butter(fs=fs,fc=fs/10,order=3) 
		self.filters['dFdt'] 	= RT_lowpass_butter(fs=fs,fc=fs/10,order=3)
		self.filters['jerk'] 	= RT_lowpass_butter(fs=fs,fc=fs/10,order=3)
		#self.filters['acc'] 	= RT_lowpass_butter(fs=fs,fc=fs/10,order=3)
		self.filters['acc'] 	= lambda tmp: tmp
		
		
		# Guards		====================
		self.t_arm_ready_ = False
		self.base_world_ready_ = False	
		self.world_arm_ready_ = False
		
		self.arm_max_vel_ = 2.0
		self.arm_max_acc_ = 2.5

		self.wait_for_transformations()
	
		
	#######################################################
	# COMPUTATIONS ########################################
	#######################################################	
	def filt_lowpass(self,data):
		fs = 1/self.dt 
		#cutoff = fs/30. #100.0
		cutoff = fs/10.
		#b,a = butter_lowpass(cutoff, fs, order=3)
		#data_filt = signal.filtfilt( b, a, data )
		data_filt = butter_lowpass_filter(data, cutoff, fs, order=3)
		return data_filt
		
		
	def wait_for_transformations(self):
		print('Getting transform')
		listener = TransformListener() 
		print('|\t Waiting for TF from: base_link to: wrist_3_link')
		while self.get_rotation_matrix(listener,self.base_link,self.end_link) is not None:
			rospy.sleep(0.5)
			if rospy.is_shutdown(): break
			
		print('|\t Waiting for arm state...')
		while np.all(self.current_position==0):
			rospy.sleep(0.5)
			if rospy.is_shutdown(): break
		
		self.t_arm_ready_ = True
		self.base_world_ready_ = True
		self.world_arm_ready_ = True 
		print('|\t The Force/Torque sensor is ready to use')
		
		self.state['pos'] = 0
		self.state['vel'] = 0
		self.state['acc'] = 0
		self.state['jerk'] = 0
		self.state['F'] = 0
		self.state['dFdt'] = 0
		self.state['dt'] = 0
		
		
		self.state['pos'] = self.current_position[0]
		self.state['vel'] = self.arm_desired_twist_adm_[0]
		self.state['acc'] = 0
		self.state['F'] = self.current_ext_wrench_vec[0]
		self.state['dt'] = time.time()
		self.last_sample_time = time.time()
		
	def get_rotation_matrix(self,listener,from_frame,to_frame,check=False):
		try:
			t = listener.getLatestCommonTime(from_frame,to_frame) # "/base_link", "/map"
			position, quaternion = listener.lookupTransform(from_frame,to_frame, t)
			rot_mat = quaternion_matrix(quaternion)[:3,:3]
			if check: return True
			else: return rot_mat 
		except Exception as e:
			#print(f'|\t Waiting for transform {e}')
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
		current_orientation_quat = self. current_orientation_quat
		desired_orientation_quat = np.array(self.desired_pose[3:])
		desired_position = np.array(self.desired_pose[:3])
		arm_desired_twist_adm_ = self.arm_desired_twist_adm_
		wrench_external_ = self.current_ext_wrench_vec
		D_ = self.D
		M_ = self.M
		K_ = self.K
		
		# Check timestep ------------------------------------------------- 
		dt = time.time() - self.last_sample_time
		#if dt < 0.5*self.dt or dt > 1.5*self.dt: return None
		dt = max(dt,0.5*self.dt)
		dt = min(dt,1.5*self.dt)
		
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
		arm_desired_accelaration[0] = self.filters['acc'](arm_desired_accelaration[0])
		"""
		n_window = 6
		discount = np.arange(n_window)/np.sum(np.arange(n_window))
		if len(self.hist['acc'])>n_window:
			acc_samples = np.array(self.hist['acc'][-(len(discount)-1):] + [arm_desired_accelaration[0]])
			acc_filt = np.sum(discount*acc_samples)
			arm_desired_accelaration[0] = acc_filt
			#acc_filt_fact = 0.1
			#acc_new = arm_desired_accelaration[0]
			#acc_last = self.state['acc']
			#acc_filt = (1-acc_filt_fact)*acc_last + (acc_filt_fact)*acc_new
			#arm_desired_accelaration[0] = acc_filt
		"""
		
		# Integrate for velocity based interface rospy.Rate --------------- 
		self.arm_desired_twist_adm_ += arm_desired_accelaration * dt # self.dt
		
		# Calculate derivatives ---------------------------------------------
		dFdt = (self.state['F'] - wrench_external_[0])/dt #self.dt
		jerk = (self.state['acc'] - arm_desired_accelaration[0])/dt #self.dt
		
		# Update Current state for RL -------------------------------------- 
		self.state['dFdt'] = self.filters['dFdt'](dFdt)
		self.state['jerk'] = self.filters['dFdt'](jerk)
		self.state['acc'] = arm_desired_accelaration[0]
		self.state['vel'] = arm_desired_twist_adm_[0]
		self.state['pos'] = current_position[0]
		
		self.state['F'] = wrench_external_[0]
		self.state['dt'] = dt # time.time() - self.last_sample_time 
		self.last_sample_time = time.time()
		for key in self.max_state.keys(): self.max_state[key] = max(self.max_state[key],abs(self.state[key]))
		for key in self.hist.keys(): self.hist[key].append(self.state[key])
		
		# Report --------------------------------------------------------------
		print(f'\nCalculate_admittance {len(self.hist["F"])}')
		print(f'|\t v[{len(self.hist["vel"])}]]    : {round(self.state["vel"],2)} \t {round(self.max_state["vel"],2)}')
		print(f'|\t dvdt : {round(self.state["acc"],2)} \t {round(self.max_state["acc"],2)}')
		print(f'|\t F    : {round(self.state["F"],2)} \t {round(self.max_state["F"],2)}')
		print(f'|\t dFdt : {round(self.state["dFdt"],5)} \t {round(self.max_state["dFdt"],5)}')
		print(f'|\t J    : {round(self.state["jerk"],5)} \t {round(self.max_state["jerk"],5)}')
		print(f'|\t dt    : {round(self.state["dt"],5)} \t {round(self.max_state["dt"],5)}')



		"""
		print(f'\ncalculate_admittance')
		print(f'|\t pose (curr):        {[list(current_position.round(2)), list(current_orientation_quat.round(2))]}')
		print(f'|\t pose (des):         {[list(desired_position.round(2)), list(desired_orientation_quat.round(2))]}')
		print(f'|\t error:     		  {self.error.round(2)}')
		print(f'|\t wrench (arm):	  {coupling_wrench_arm.round(2)}')
		print(f'|\t wrench (ext)[{np.argmax(wrench_external_[:3])}]:	  {wrench_external_.round(2)}')
		print(f'|\t acc_des: 		  {arm_desired_accelaration.round(2)}')
		print(f'|\t twist_des: 		  {self.arm_desired_twist_adm_.round(2)}')
		"""
		
	
		
	def send_command(self):
		scale = 1.0 
		# Check limits ------------------
		v_norm = np.linalg.norm(self.arm_desired_twist_adm_[:3])
		if v_norm > self.arm_max_vel_: self.arm_desired_twist_adm_[:3] *= (self.arm_max_vel_ / v_norm)
		v_norm = np.linalg.norm(self.arm_desired_twist_adm_[3:])
		if v_norm > self.arm_max_vel_: self.arm_desired_twist_adm_[3:] *= (self.arm_max_vel_ / v_norm)
		
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
		# Settings --------------------------
		torque_deadzone_lower_limit = 2.0
		force_deadzone_lower_limit = 3.0
		force_deadzone_upper_limit = 100.0
		force_scale = 1.0 
		filt_factor = 0.8

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
		"""
		n_window = 1000
		#wrench_new[0] = self.filt_lowpass(self.hist['F']+ [wrench_new[0]])[-1]
		#discount = np.arange(n_window)/np.sum(np.arange(n_window))
		#discount = np.ones(n_window)/np.sum(n_window)
		if len(self.hist['F'])>n_window:
			force_samples = np.array(self.hist['F'][-(n_window-1):] + [wrench_new[0]])
			
			# Move Ave Fulter
			#F_filt = np.sum(discount*force_samples)
			#wrench_new[0] = F_filt
			
			# Butter Fulter
			wrench_new[0] = self.filt_lowpass(force_samples)[-1]
		"""
		# Check bounds and errors ---------------------------------------
		for i in range(3):
			# lower force bound
			if abs(wrench_new[i]) < force_deadzone_lower_limit: wrench_new[i] = 0
			elif wrench_new[i] >0: wrench_new[i] -= force_deadzone_lower_limit
			else: wrench_new[i] += force_deadzone_lower_limit
		
			# upper force bound
			if wrench_new[i] >0: wrench_new[i] = min(abs(wrench_new[i]),force_deadzone_upper_limit)
			else: wrench_new[i] = - min(abs(wrench_new[i]),force_deadzone_upper_limit)
			
			# lower torque bound
			if abs(wrench_new[i+3]) < torque_deadzone_lower_limit: wrench_new[i+3] = 0
			elif wrench_new[i+3] >0: wrench_new[i+3] -= torque_deadzone_lower_limit
			else: wrench_new[i+3] += torque_deadzone_lower_limit
		wrench_new = np.nan_to_num(wrench_new)
			
		# Update wrench
		self.current_ext_wrench_vec = wrench_new
		
	#######################################################
	# CONTROLLER HANDLERS #################################
	#######################################################	
	def get_state(self):
		v = self.state['vel']
		dvdt = self.state['acc'] 
		F = self.state['F']
		dFdt =self.state['dFdt'] 
		J = self.state['jerk']
		return v,dvdt,F,dFdt,J 
		
	def set_damping(self,D):
		D = max(min(D,self.D_limits[1]),self.D_limits[0])
		self.D[0,0] = D
		return D
		
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
			self.compute_admittance()
			cmd = self.send_command() 
			self.loop_rate.sleep()
		#self.stop_all()
		#self.loop_rate.sleep()
		
		nkeys = len(list(self.hist.keys()))
		fig,axs = plt.subplots(nkeys,1)
		for i, key in enumerate(self.hist.keys()):
			axs[i].plot(self.hist[key])
			"""
			if key == 'jerk':
				yfilt = self.filt_lowpass(self.hist[key])
				axs[i].plot(yfilt)
			elif key == 'F':
				yfilt = self.filt_lowpass(self.hist[key])
				axs[i].plot(yfilt)
			elif key == 'dFdt':
				yfilt = self.filt_lowpass(self.hist[key])
				axs[i].plot(yfilt)
			"""
			axs[i].set_ylabel(key)
		plt.show()
			
	def step(self):
		print('Stepping Controller...')
		if not rospy.is_shutdown():
			self.compute_admittance()
			cmd = self.send_command() 
			self.loop_rate.sleep()
			
if __name__=="__main__":
	controller = Admittance() 
	
	controller.run()
		
def bound(val,bnds,test=False):
	if test:return (val>=bds[0] and val <= bnds[1])
	return min(max(val,bnds[0]),bnds[1])

		
