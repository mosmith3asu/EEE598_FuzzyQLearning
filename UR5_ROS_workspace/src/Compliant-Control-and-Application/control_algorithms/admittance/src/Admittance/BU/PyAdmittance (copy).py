#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Pose,Twist, Wrench
from geometry_msgs.msg import WrenchStamped 
from cartesian_state_msgs.msg import PoseTwist
from tf import TransformListener, listener
from std_msgs.msg import String
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler
#from time import time
# Twist: this expresses velocity in free space broken into it's linear and angular parts. 
# | linear.{x,y,z}
# | angular.{x,y,z}
#pub_arm_cmd_  = nh_.advertise<geometry_msgs::Twist>(topic_arm_command, 5);

"""
// ROS VARIABLES:
  ros::NodeHandle nh_;
  ros::Rate loop_rate_;

  // ADMITTANCE PARAMETERS:
  Matrix6d M_, D_, K_;

  // Subscribers:
  ros::Subscriber sub_arm_state_;
  ros::Subscriber sub_wrench_state_;
  // Publishers:
  ros::Publisher pub_arm_cmd_;

  // Variables:
  Vector3d	  arm_position_;
  Quaterniond   arm_orientation_;
  Vector6d	  arm_twist_;
  Vector6d	  wrench_external_;
  Vector6d	  arm_desired_twist_adm_;
  Vector6d	  arm_desired_accelaration;

  Vector7d	  desired_pose_;
  Vector3d	  desired_pose_position_;
  Quaterniond   desired_pose_orientation_;

  Vector6d	  error;

  // TF:
  // Transform from base_link to world
  Matrix6d rotation_base_;
  // Listeners
  tf::TransformListener listener_ft_;
  tf::TransformListener listener_control_;
  tf::TransformListener listener_arm_;

  // Guards
  bool ft_arm_ready_;
  bool base_world_ready_;
  bool world_arm_ready_;

  double arm_max_vel_;
  double arm_max_acc_;
"""
"""
		self.M = np.array([
			[5,0,0,0,0,0],
          	[0,5,0,0,0,0],
          	[0,0,5,0,0,0],
          	[0,0,0,5,0,0],
          	[0,0,0,0,5,0],
          	[0,0,0,0,0,5]
		])
		self.D = np.array([
           	[5,0,0,0,0,0],
           	[0,5,0,0,0,0],
           	[0,0,5,0,0,0],
           	[0,0,0,5,0,0],
           	[0,0,0,0,5,0],
          	[0,0,0,0,0,5]
          ])
          """
class Admittance(object):
	def __init__(self):
		# PARAMS		====================
		print('Initializing Admittance Object...')
		self.base_link = "/base_link"
		self.end_link = "/wrist_3_link"
		self.topic_arm_command = '/cartesian_velocity_controller/command_cart_vel'
		self.topic_arm_state = '/cartesian_velocity_controller/ee_state'
		self.topic_wrench_state = '/wrench'
		self.node_handler = 'admittance_controller'
		
		self.desired_position = [0.0,0.3,0.7]
		rpy = [0.0,-1.5707,0.0] #[90deg,0,-90deg]
		self.desired_orientation_quat = quaternion_from_euler(0.0,1.5707,0.0)#[0.0, 0.0, 0.707, -0.707]
		self.desired_pose = self.desired_position + list(self.desired_orientation_quat )
		#self.desired_pose = [0.0,0.3,0.7,       0.0, 0.0, 0.707, -0.707]
		self.workspace_limits: [-0.50, 0.50, 0.25, 0.80, 0.30, 0.75]
		
		self.M = np.diag([5,5,5,5,5,5])
		self.D = np.diag([5,50,50,50,50,50])
		self.K = np.diag([10.,100.,100.,100.,100.,100.])

		
		# Init Node 	====================
		self.dt = 0.1
		rospy.init_node(self.node_handler, anonymous=True)
		self.loop_rate = rospy.Rate(1/self.dt)
		
		# Subscribers 	====================
		self.sub_arm_state_ = rospy.Subscriber(self.topic_arm_state, PoseTwist, self.state_arm_callback)
		self.sub_wrench_state_ = rospy.Subscriber(self.topic_wrench_state, WrenchStamped, self.state_wrench_callback)
		
		# Publisher	====================		
		self.pub_arm_cmd_ = rospy.Publisher(self.topic_arm_command, Twist, queue_size=5)
		
		# Initializing the class variables: ====================
		self.rot_matrix = np.zeros([6,6])
		self.arm_position_ = np.zeros(3) 			# Vector3D
		self.arm_orientation_ = np.zeros(4) 		# Quaternion
		self.arm_twist_ = np.zeros(6) 			# Vector6D
		self.wrench_external_= np.zeros(6)  		# Vector6D
		self.arm_desired_twist_adm_ = np.zeros(6) 	# Vector6D Twist()
		self.arm_desired_accelaration_ = np.zeros(6) # Vector6D
		self.current_pose = Pose()
		self.current_twist = Twist()
		self.current_position = np.zeros(3)
		self.current_orientation_quat = np.zeros(4)
		self.current_ext_wrench = Wrench()
		self.current_ext_wrench_vec = np.zeros(6)
		
		
		self.desired_pose_position_= np.zeros(7) 	# Vector7D
		#self.desired_pose_position_= np.zeros(3) 	# Vector3D
		#self.desired_pose_orientation_ = np.zeros(4) # Quaternion
		
		self.error = np.zeros(6) 				# Vector6D
	
		# TF: Transform from base_link to world ====================
		self.rotation_base_ = np.zeros([6,6])
		
		# Listeners	====================
		self.listener_ft_ = TransformListener()
		self.listener_control_ = TransformListener()
		self.listener_arm_ = TransformListener()
		
		# Guards		====================
		self.t_arm_ready_ = False
		self.base_world_ready_ = False
		self.world_arm_ready_ = False
		
		self.arm_max_vel_ = 0.1
		self.arm_max_acc_ = 0.2
		self.force_x_pre = 0
		self.force_y_pre = 0
		self.force_z_pre = 0
		
		print('Getting transform')
		self.wait_for_transformations()
		
		
	def wait_for_transformations(self):
		listener = TransformListener()
		#rospy.loginfo('Getting transform')
		while not self.get_rotation_matrix(self.rot_matrix,listener,self.base_link,self.end_link):
			#print('waiting')
			rospy.sleep(0.5)
		self.t_arm_ready_ = True
		self.base_world_ready_ = True
		self.world_arm_ready_ = True
		#rospy.loginfo(rospy.get_caller_id() + 'The Force/Torque sensor is ready to use')
		print('|\t The Force/Torque sensor is ready to use')
		
		
	def get_rotation_matrix(self,rotation_matrix,listener,from_frame,to_frame):
		try:
			t = listener.getLatestCommonTime(from_frame,to_frame) # "/base_link", "/map"
			position, quaternion = listener.lookupTransform(from_frame,to_frame, t)
			#rotation_matrix
			#print position, quaternion
			print('|\t Got transform!')
			return True
		except:
			print('|\t Waiting for transform')
			return False
		
		
	def compute_admittance(self):
		def quaternion_error(q1,q2): 
			#q1 = np.array(q1).reshape([1,4])
			#q2_inv = np.array(q2).reshape([1,4])
			q2_inv = np.copy(q2)
			q2_inv[3] = - q2_inv[3] 
			return q1*q2_inv
		def quaternion2euler(q): 
			return euler_from_quaternion(q)
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
		
		# Translation error w.r.t. desired equilibrium
		self.error[0:3] = current_position -desired_position
		
		# Rotation error w.r.t. desired equilibrium
		error_quat = quaternion_error(current_orientation_quat,desired_orientation_quat)
		error_quat = error_quat/np.linalg.norm(error_quat)
		self.error[3:] = quaternion2euler(error_quat)
		
		# Convert to axis angles err_arm_des_orient.axis() * err_arm_des_orient.angle()
		coupling_wrench_arm = D_*(arm_desired_twist_adm_) + K_ * self.error
		arm_desired_accelaration = np.linalg.inv(M_) * ( - coupling_wrench_arm  + wrench_external_)
		arm_desired_accelaration = np.array(np.diagonal(arm_desired_accelaration))
		a_acc_norm = np.linalg.norm(arm_desired_accelaration[:3])
		if a_acc_norm > self.arm_max_acc_:
			arm_desired_accelaration[:3] *= (self.arm_max_acc_/a_acc_norm)
		
		
		
		# Integrate for velocity based interface rospy.Rate
		#dt = self.loop_rate.sleep_dur #.expectedCycleTime()
		self.arm_desired_twist_adm_ += arm_desired_accelaration * self.dt
		print(f'\ncalculate_admittance')
		print(f'|\t pose (curr):        {[list(current_position.round(2)), list(current_orientation_quat.round(2))]}')
		print(f'|\t pose (des):         {[list(desired_position.round(2)), list(desired_orientation_quat.round(2))]}')
		print(f'|\t error:     		  {self.error.round(2)}')
		print(f'|\t acc_des: 		  {arm_desired_accelaration.round(2)}')
		print(f'|\t twist_des: 		  {self.arm_desired_twist_adm_.round(2)}')
		
		
		
		
	def send_command(self):
		scale = 1.0#0.3
		arm_twist_cmd = Twist()
		
		# Check limits
		v_norm = np.linalg.norm(self.arm_desired_twist_adm_[:3])
		if v_norm > self.arm_max_vel_:
			self.arm_desired_twist_adm_[:3] *= (self.arm_max_vel_ / v_norm)
		
		arm_twist_cmd.linear.x  = self.arm_desired_twist_adm_[0]*scale
		arm_twist_cmd.linear.y  = self.arm_desired_twist_adm_[1]*scale
		arm_twist_cmd.linear.z  = self.arm_desired_twist_adm_[2]*scale
		arm_twist_cmd.angular.x = self.arm_desired_twist_adm_[3]*scale
		arm_twist_cmd.angular.y = self.arm_desired_twist_adm_[4]*scale
		arm_twist_cmd.angular.z = self.arm_desired_twist_adm_[5]*scale
		self.pub_arm_cmd_.publish(arm_twist_cmd)
		return arm_twist_cmd
		
		
	# CALLBACKS ##########################################3
	def state_arm_callback(self,msg):
		#rospy.loginfo(data.data)
		# type: cartesian_state_msgs:PoseTwistConstPtr
		#print(msg)
		#rospy.loginfo(msg.pose)
		self.current_pose.position.x = msg.pose.position.x
		self.current_pose.position.y = msg.pose.position.y
		self.current_pose.position.z = msg.pose.position.z
		self.current_pose.orientation.x = msg.pose.orientation.x
		self.current_pose.orientation.y = msg.pose.orientation.y
		self.current_pose.orientation.z = msg.pose.orientation.z
		self.current_pose.orientation.w = msg.pose.orientation.w
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
		
		self.current_twist.linear.x = msg.twist.linear.x
		self.current_twist.linear.y = msg.twist.linear.y
		self.current_twist.linear.z = msg.twist.linear.z
		self.current_twist.angular.x = msg.twist.angular.x
		self.current_twist.angular.y = msg.twist.angular.y
		self.current_twist.angular.z = msg.twist.angular.z
		
		
	def state_wrench_callback(self,msg):
		#rospy.loginfo(msg)
		#force_thres_lower_limit_ = 50
		#force_thres_upper_limit_ = 100
		
		self.current_ext_wrench.force.x = msg.wrench.force.x
		self.current_ext_wrench.force.y = msg.wrench.force.y
		self.current_ext_wrench.force.z = msg.wrench.force.z
		self.current_ext_wrench.torque.x = msg.wrench.torque.x
		self.current_ext_wrench.torque.y = msg.wrench.torque.y
		self.current_ext_wrench.torque.z = msg.wrench.torque.z
		
		self.current_ext_wrench_vec[0] = msg.wrench.force.x
		self.current_ext_wrench_vec[1] =  msg.wrench.force.y
		self.current_ext_wrench_vec[2] = msg.wrench.force.z
		self.current_ext_wrench_vec[3] =  msg.wrench.torque.x
		self.current_ext_wrench_vec[4] = msg.wrench.torque.y
		self.current_ext_wrench_vec[5] =  msg.wrench.torque.z

		
		#self.current_ext_wrench = msg.wrench
		#force_bounds = [force_thres_lower_limit_,force_thres_upper_limit_]
		#"""geometry_msgs::WrenchStampedConstPtr"""
		#if bound(wrench_ft_frame[0],force_bounds,test=True):
		#	pass
		#else:
		#	wrench_ft_frame[0] = 0
	def run(self):
		print('Running Controller...')
		while not rospy.is_shutdown():
			self.compute_admittance()
			cmd = self.send_command() #'|\t sending...' +
			#print(cmd)
			#rospy.spin()
			#print(self.current_pose)
			self.loop_rate.sleep()
if __name__=="__main__":
	controller = Admittance() 
	
	controller.run()
		
def bound(val,bnds,test=False):
	if test:return (val>=bds[0] and val <= bnds[1])
	return min(max(val,bnds[0]),bnds[1])

		
