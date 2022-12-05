#!/usr/bin/env python3
#from PyAdmittance import Admittance
from RL_ROS import RL
import rospy
import numpy as np
from std_msgs.msg import Float32,String,Bool,UInt16
#import json
import ast
import time
import matplotlib.pyplot as plt
from MJT import get_MJT
from datetime import datetime
import copy
from Controller_Config import ALL_SETTINGS
SETTINGS = ALL_SETTINGS.learning_node
FILTERS = ALL_SETTINGS.filters
TOPICS = ALL_SETTINGS.topics
PUBLIC = ALL_SETTINGS.public

if __name__=="__main__":
	plt.ion()
	plt_titles = ['Episode Reward','Episode Length','Episode Energy','Damping','Velocity','Force']
	plt_xlims = [[0,SETTINGS['num_episodes']],[0,SETTINGS['num_episodes']],[0,50],[-2,2]]
	fig,axs = plt.subplots(len(plt_titles),1)
	_t = np.linspace(0,1,10)
	_y = np.linspace(0,1,10)
	lines =[]
	
	#start2goal_cm = [0,57] # cm
	#start2goal_ros = [-0.3,0.3]
	
	pos2dist = lambda _pos: (57./0.6)*(_pos+0.3)
	
	print(f'Initializing plots...')
	for i,name in enumerate(plt_titles):
		#if name == 'Velocity":
		#	l, = axs[i].plot(_t,_y)
		print(f'\t| [i]{name}')
		l, = axs[i].plot(_t,_y)
		axs[i].set_ylabel(name)
		#axs[i].set_ylim(plt_xlims[i])
		lines.append(l)
		
		
	fig.canvas.flush_events()
	fig.canvas.draw()


class MemoryHandler(object):
	def __init__(self):
		self.sticky_names = ['Episode Reward','Episode Length','Episode Energy']
		self.nonsticky_names = ['Damping','Velocity','Force'] # 'Jerk'
		self.names = self.sticky_names +self.nonsticky_names
		
		now = datetime.now()
		self.date_label = now.strftime("-Date_%m_%d_%Y-Time_%H_%M_%S")
		self.fname = 'Data_' + self.date_label + '.npz'
		
		
		#start2goal_cm = [0,57] # cm
		#start2goal_ros = [-0.3,0.3]
		self.pos2dist = lambda _pos: (57./0.6)*(_pos+0.3)
		
		self.epi_data = []
		
		
		self.data = {}
		for name in self.names: self.data[name] = []
	def geti(self,name): return self.names.index(name)
	def get(self,name): return self.data[name]
	def add(self,name,value): self.data[name].append(value)
	def replace(self,name,value): self.data[name][-1] = value
	def reset(self):
		# export to perminant buffer
		#pckg = copy.deepcopy(self.epi_data_pckg)
		#for key in names: pckg[key] = self.data[name]
		#self.epi_data.append(copy.deepcopy(pckg))
		self.epi_data.append(copy.deepcopy(self.data))
		for name in self.nonsticky_names:
			self.data[name] = []
	def save(self):
		print(f'SAVING DATA AS [{self.fname}] [nEpi={len(self.epi_data)}]')
		np.savez_compressed(self.fname ,a=self.epi_data)
	def load(self,fname):
		plt.ioff()
		
		#np.savez_compressed(self.fname ,a=self.epi_data)
		loaded = np.load(fname,allow_pickle=True)
		self.epi_data = loaded['a']
		
		
		data = self.epi_data[-1]
		print(f'LOADING DATA AS [{fname}]')
		print(f'\t| N episodes: {len(self.epi_data)}')
		print(f'\t| Data Types:')
		#for name in data.keys(): 
		for name in self.names: 
			is_missing = (name not in data.keys())
			if is_missing: print(f'\t\t -[MISSING] {name}')
			else: print(f'\t\t - {name}')
		

		
		
		T = data['Episode Length']
		Ft_epif = data['Force']
		t_epif = np.linspace(0,T,len(Ft_epif))
		dt = t_epif[1]
		begin_idx = np.array(np.where(F>0)).flatten()[0]
		
		
		CumReward_epif = data['Episode Reward']
		Length_epif = np.array(data['Episode Length']) - dt*begin_idx
		Energy_epif = data['Episode Energy'][begin_idx:]
		Dt_epif = data['Damping'][begin_idx:]
		vt_epif = data['Velocity'][begin_idx:]
		Ft_epif = Ft_epif[begin_idx:]
		
		
		
		# Calculation of Dynamics				
		pt_epif = np.zeros(len(t_epif))
		at_epif = np.zeros(len(t_epif))
		Jt_epif = np.zeros(len(t_epif))
		
		for i in range(1,len(t_epif)):
			pt_epif[i] = pt_epif[i] + v_epif[i-1]*dt
		at_epif = np.gradient(v_epif)/dt
		Jt_epif = np.gradient(v_epif)/dt
		xt_MJT,vt_MJT,at_MJT,Jt_MJT = get_MJT(tf = t_epif[-1],xf = pt_epif,N= len(t_epif))
		
		
		
		stats = [CumReward_epif,Length_epif,Energy_epif]
		epif = [pt_epif,vt_epif,at_epif,Jt_epif]
		MJT = [xt_MJT,vt_MJT,at_MJT,Jt_MJT]
		return stats,epif,MJT

		
		#nVars = len(data.keys())
		#fig,axs = plt.subplots(nVars,1)
		#for i,name in enumerate(data.keys()): 
		#	axs[i].plot(data[name],label=name)
		#	axs[i].set_ylabel(name)
		#	#if name== 'Velocity':
		
			
		plt.show()

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
	

def main():
	# RL Settings
	ENABLE_PLOT = True
	FIX_DAMPING = False
	
	exp_fs = SETTINGS['loop_rate']						# expected controller sample freq
	exp_dt = 1/exp_fs										# expected controller timestep
	rospy.init_node(SETTINGS['node_handler'], anonymous=True) 	# start ROS node
	loop_rate = rospy.Rate(exp_fs)					# init ROS node freq
	t_last_sample = time.time()
	
	
	# Controller Settings
	goal_position = PUBLIC['pos_des']
	start_position = PUBLIC['pos_start']
	last_position = start_position
	
	max_epi_duration = SETTINGS['max_epi_duration'] # seconds
	num_episodes = SETTINGS['num_episodes']
	tol = PUBLIC['tol_des']
	n_step_update =  SETTINGS['n_step_update']
	

	print(f'Initializing virtual controller...')
	controller = VirtualController()
	controller.load_settings(goal_position,max_epi_duration,tol)
	D_DEFAULT = 10.
	
	
	memory = MemoryHandler()

	Energy = 0
	v0 = 0.0 
	v_dot0 = 0.0 
	F0 = 0.0 
	F_dot0 = 0.0 
	Yousef = RL(v0,v_dot0,F0,F_dot0)
	
	reward_list=[]
	
	for ith_episode in range(num_episodes):
		if rospy.is_shutdown(): break
		print(f'\n\n\nStarting Epi={ith_episode}')
		controller.reset()

		v,dvdt,F,dFdt,Jerk0 = controller.get_state()
		Yousef.__init__(v,dvdt,F,dFdt)
		d,r=Yousef.apply_first_action(v,dvdt,F,dFdt,Jerk0)
		#d = FILTERS['damping'].sample(d)
		
		comu_reward=r;
		t_last_sample = time.time()
		# ['Episode Reward','Episode Length','Damping','Velocity']
		
		memory.add('Episode Reward',comu_reward)
		memory.add('Episode Length', 0)
		memory.add('Episode Energy', 0)
		memory.add('Damping', d)
		memory.add('Velocity',Jerk0)
		memory.add('Force',F)
		#memory.add('Jerk',Jerk0)
		
		memory.reset()
		memory.save()
		Jmemory = []
		
		
		#if FIX_DAMPING: d=D_DEFAULT ## DUBUG ##
		#controller.set_damping(d)
		
		if FIX_DAMPING: controller.set_damping(D_DEFAULT) ## DUBUG ##
		else: controller.set_damping(d)
		
		
		#watchdog = {'start': time.time(), 'trigger': 1.1*max_epi_duration}
		#while controller.state['timestamp'] < max_epi_duration:
		while not controller.done:
			if rospy.is_shutdown(): break
			print(f"\r iter: t={controller.state['timestamp']} Rc = {comu_reward}",end='')
			# GET CURRENT STATE 
			v,dvdt,F,dFdt ,Jerk = controller.get_state()
			Jmemory.append(Jerk) #Jerk0=Jerk0+Jerk
			
			# CHECK TERMINAL CONDITION
			if controller.done: break
			
			####################################################
			# SAMPLE RL ALGORITHM @ LOWER FREQ ##################
			if len(Jmemory) >= n_step_update: # EVERY 10 ITER
				#Jerk0=Jerk0/10 #\n d,r=Yousef.apply_action(v,dvdt,F,dFdt,r,Jerk0)
				Jc = np.mean(Jmemory)
				#Jc = np.linalg.norm(Jmemory,ord=np.inf) # 
				d,r=Yousef.apply_action(v,dvdt,F,dFdt,r,Jc)
				#d = FILTERS['damping'].sample(d)
				comu_reward=comu_reward+r
				Jmemory = [] # Jerk0=0
				
				
				
				######################################################
			# ACT ACCORDING TO POLICY
			
			reward_list.append(comu_reward)
			if FIX_DAMPING: controller.set_damping(D_DEFAULT) ## DUBUG ##
			else: controller.set_damping(d)
			
			d,_r = Yousef.select_action(v,dvdt,F,dFdt,Jerk)
			#d = FILTERS['damping'].sample(d)
			
			###### ADD ENERGY CALC ###########
			dx = pos2dist(controller.state['pos'])-last_position
			Energy = Energy + abs(F)*dx
			
			# Record timestep
			memory.replace('Episode Reward', comu_reward)
			memory.replace('Episode Length', controller.state['timestamp'] )
			memory.replace('Episode Energy', Energy)
			memory.add('Damping', d)
			memory.add('Velocity',v)
			memory.add('Force',F)
			#memory.add('Jerk',Jerk)
			
			
	
			loop_rate.sleep()
		################# END EPISODE ########
		#"""
		now = memory.get('Episode Length')
					
		iax = memory.geti('Damping')
		ydata = memory.get('Damping')		
		xdata = np.arange(len(ydata))
		line = lines[iax]
		line.set_xdata(xdata)
		line.set_ydata(ydata)
		print(f'Damping: [{len(xdata)},{len(ydata)}]')
		line.axes.set_ylim([min(ydata), max(ydata)])
		line.axes.set_xlim([min(xdata), max(xdata)])
					
		iax = memory.geti('Velocity')
		ydata = memory.get('Velocity')
		xdata = np.arange(len(ydata))
		line = lines[iax]
		line.set_xdata(xdata)
		line.set_ydata(ydata)
		print(f'Velocity: [{len(xdata)},{len(ydata)}]')
		line.axes.set_ylim([min(ydata), max(ydata)])
		line.axes.set_xlim([min(xdata), max(xdata)])
		
		iax = memory.geti('Force')
		ydata = memory.get('Force')
		xdata = np.arange(len(ydata))
		line = lines[iax]
		line.set_xdata(xdata)
		line.set_ydata(ydata)
		print(f'Velocity: [{len(xdata)},{len(ydata)}]')
		line.axes.set_ylim([min(ydata), max(ydata)])
		line.axes.set_xlim([min(xdata), max(xdata)])
		"""
		iax = memory.geti('Jerk')
		ydata = memory.get('Jerk')
		xdata = np.arange(len(ydata))
		line = lines[iax]
		line.set_xdata(xdata)
		line.set_ydata(ydata)
		line.axes.set_ylim([min(ydata), max(ydata)])
		line.axes.set_xlim([min(xdata), max(xdata)])
		"""
		
		iax = memory.geti('Episode Reward')
		ydata = memory.get('Episode Reward')
		xdata = np.arange(len(ydata))
		line = lines[iax]
		line.set_xdata(xdata)
		line.set_ydata(ydata)
		line.axes.set_ylim([min(ydata), max(ydata)])
		line.axes.set_xlim([min(xdata), max(xdata)])
		
		iax = memory.geti('Episode Length')
		ydata = memory.get('Episode Length')
		xdata = np.arange(len(ydata))
		line = lines[iax]
		line.set_xdata(xdata)
		line.set_ydata(ydata)
		line.axes.set_ylim([min(ydata), max(ydata)])
		line.axes.set_xlim([min(xdata), max(xdata)])
		
		
		iax = memory.geti('Episode Energy')
		ydata = memory.get('Episode Energy')
		xdata = np.arange(len(ydata))
		line = lines[iax]
		line.set_xdata(xdata)
		line.set_ydata(ydata)
		line.axes.set_ylim([min(ydata), max(ydata)])
		line.axes.set_xlim([min(xdata), max(xdata)])
		
		
		fig.canvas.flush_events()
		fig.canvas.draw()
		
		
		
		
		
		
			
if __name__=="__main__":
	main()		
