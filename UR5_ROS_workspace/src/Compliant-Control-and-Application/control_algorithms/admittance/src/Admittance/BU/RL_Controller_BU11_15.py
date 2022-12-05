#!/usr/bin/env python3
from PyAdmittance import Admittance
from RL_ROS import RL
import rospy
import numpy as np
from std_msgs.msg import Float32
"""
def check_terminal(tau,v,dvdt,F,dFdt):
	#Check consequtive terminal states for final condition
	thresh = 0.1
	is_terminal = [(tau>10)]
	is_terminal.append((abs(F) < thresh))
	is_terminal.append((abs(dFdt) < thresh))
	#for arg in args: is_terminal.append(arg < thresh)
		
	return np.all(is_terminal)
"""

if __name__=="__main__":
	#pub = rospy.Publisher('RO_damping_cmd', Float32, queue_size=10)
	#rospy.init_node('RL_damping', anonymous=True)
	#rate = rospy.Rate(100) 
	
	# Settings
	START_POSITION = -0.4
	GOAL_POSITION = 0.0
	num_episodes = 3
	max_epi_duration = 30. # seconds
	tau2t = 10.
	tau = 0 
	dtau = 0.005
	dt = tau2t*dtau
	
	# Intiate objects
	
	controller = Admittance(dtau = dtau) 
	terminal = {}
	terminal['pos'] = GOAL_POSITION # goal position
	terminal['vel'] = 0.0 # goal velocity
	terminal['acc'] = 0.0 # goal acceleration
	terminal['F'] = 0.0 # human satisfied with position
	terminal['dFdt'] = 0.0 # human satisfied with position
	terminal['tol'] = 0.1 # maximum error considered @ goal
	terminal['max_duration'] = max_epi_duration # seconds
	controller.set_terminal(terminal)
	

	v0 = 0.0#controller.state['vel']
	v_dot0 = 0.0#controller.state['acc']
	F0 = 0.0#controller.state['F']
	F_dot0 = 0.0#controller.state['dFdt']
	Yousef = RL(v0,v_dot0,F0,F_dot0)
	
	START_POSITION = controller.state['pos']
	reward_list=[]
	Jerk0 = 0
	for ith_episonde in range(num_episodes):
		"""m
		v,dvdt,F,dFdt,Jerk0 = controller.get_state()
		Yousef.__init__(v,dvdt,F,dFdt)
		d,r=Yousef.apply_first_action(v,dvdt,F,dFdt,Jerk0)
		comu_reward=r;
		"""
		
		
		
		d = 5.0 # DELETE
		controller.set_damping(d)
		
		done = controller.check_terminal()
	
		#while tau*dtau < max_epi_duration:
		while controller.state['timestamp']< max_epi_duration:
			done = controller.check_terminal()
			# GET CURRENT STATE 
			v,dvdt,F,dFdt ,Jerk = controller.get_state()
			Jerk0=Jerk0+Jerk
			
			# CHECK TERMINAL CONDITION
			if controller.check_terminal():
				break
			"""
			# SAMPLE RL ALGORITHM @ LOWER FREQ
			if tau%tau2t==0: 
				Jerk0=Jerk0/10
				d,r=Yousef.apply_action(v,dvdt,F,dFdt,r,Jerk0)
				comu_reward=comu_reward+r
				Jerk0=0
				
				#msg = Float32()
				#pub.publish(d)
			# ACT ACCORDING TO POLICY
			reward_list.append(comu_reward)
			"""
			
			controller.set_damping(d)
			controller.step()
			controller.loop_rate.sleep()
			
			
