"""
###############################################################
###############################################################
Created By  : Mason Smith
Contact: mosmith3@asu.edu
Created Date: 12/4/22
version ='1.0'
Description:
    used in the Fuzzy-Q Learning algorithm for adaptively tuning
    impedance gains under unknown target positions
###############################################################
###############################################################
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from Simulation.modules.HC import MinJerkOptimization
class Environment(object):
    def __init__(self,desired_pos, timestep,
                 mass = 1.3, K_human=1.0, D_human = 0.1,rush_human=0.0,xdur_human=5.0,N_human=100,
                 error_thresh=0.03,max_duration=10., start_pos = 0.,**kwargs
                 ):
        """
        :param desired_pos: fina x position of ee
        :param timestep: whatever timestep the environment is being sampled at (dt/dtau)
        :param mass: mass of ee in kg
        :param error_thresh: threshold for final position to be considered terminal
        :param max_duration: maximum time before state is considered terminal (not going to reach final state)
        :param start_pos: starting x position of ee
        """
        # Parameters and Settings
        self.mass = mass
        self.timestep = timestep
        self.state_names = ['vel','acc','Fext','dFextdt']
        self._desired_pos = desired_pos
        self._start_pos = start_pos

        self.terminal_conditions = {}
        self.terminal_conditions['error_thresh'] = error_thresh
        self.terminal_conditions['max_duration'] = max_duration

        self.human = {}
        self.human['stiffness'] = K_human
        self.human['damping'] = D_human
        self.human['rush'] = rush_human
        self.human['traj_duration'] = xdur_human
        #

        # optimizer = MinJerkOptimization([start_pos,0,0], [desired_pos,0,0], [0,xdur_human], N=N_human)
        # J = optimizer.get_jerk(rush=self.human['rush'])
        # self.H_MJTdes = optimizer.get_state_trajectory(J)
        # self.H_tdes = np.linspace(0,xdur_human,N_human)

        x0 = [start_pos,0,0]
        xf = [desired_pos,0,0]
        duration = [0,xdur_human]
        self.N_human = N_human
        N = N_human
        T = duration[1] - duration[0]
        # N = int( T/timestep)
        self.xdur_human = xdur_human
        #
        optimizer = MinJerkOptimization(x0, xf, duration, N)
        J = optimizer.get_jerk(rush=self.human['rush'])
        xt = optimizer.get_state_trajectory(J)
        self.H_tdes = np.linspace(duration[0], duration[1], N_human)
        self.H_MJTdes = xt
        self.H_MJT_J = J
        # optimizer.preview()

        # Initialized Aux Variables (need to call reset() )
        self.goal_state = None
        self.full_state = None
        self.memory = None




    ####################################################################
    ## Core Functions ##################################################
    ####################################################################
    def step(self, action):
        """
        Action: damping value of robotic agent
        """
        # Unpack
        mass = self.mass
        tstep = self.timestep
        pos0 = self.full_state['pos']
        vel0 = self.full_state['vel']
        acc0 = self.full_state['acc']
        Fext0 = self.full_state['Fext']
        # dFdt0 = self.full_state['dFextdt']
        # jerk0 = self.full_state['jerk']
        tsec0 = self.full_state['time_stamp']
        #
        i_tnearest =np.abs(tsec0 - (self.H_tdes-100*tstep)).argmin()
        self.goal_state['pos'] = self.H_MJTdes[i_tnearest,0,0]

        # Get interaction/coupling forces
        Fext1 = self.get_human_force()  # Get human interaction force
        Fcpl1 = self.get_robot_force(damping=action)  # get robot interaction forces

        # Calculate dynamics
        acc1 = (1 / mass) * (Fext1 + Fcpl1)  # integral
        vel1 = vel0 + (acc1 * tstep)  # integral
        pos1 = pos0 + (vel1 * tstep)  # integral
        jerk1 = (acc1 - acc0) / tstep  # derivative of acc
        dFdt1 = (Fext1 - Fext0) / tstep  # derivative of force
        tsec1 = tsec0 + tstep  # step time

        # Update full_state
        self.full_state['pos'] = pos1
        self.full_state['vel'] = vel1
        self.full_state['acc'] = acc1
        self.full_state['Fext'] = Fext1
        self.full_state['dFextdt'] = dFdt1
        self.full_state['jerk'] = jerk1
        self.full_state['time_stamp'] = tsec1
        self.full_state['action'] = action

        # Return t+1 parameters
        reward = jerk1
        obs = self.make_observation()
        done = self.check_done()
        info = None
        return obs, reward, done, info

    def reset(self):
        self.goal_state = {}
        self.goal_state['pos'] = self._desired_pos
        self.goal_state['vel'] = 0.
        self.goal_state['acc'] = 0.
        self.goal_state['Fext'] = 0.
        self.goal_state['dFextdt'] = 0.
        self.goal_state['jerk'] = 0.

        self.full_state = {}
        self.full_state['pos'] = self._start_pos
        self.full_state['vel'] = 0.
        self.full_state['acc'] = 0.
        self.full_state['Fext'] = 0.
        self.full_state['dFextdt'] = 0.
        self.full_state['jerk'] = 0.
        self.full_state['time_stamp'] = 0.
        self.full_state['action'] = 0

        self.memory_clear()
        self.memory_update()
        obs = self.make_observation()
        return obs

    def render(self,epi_rewards=None):
        clip = 3
        nRows,nCols = 4,2
        fig,axs = plt.subplots(nRows,nCols)
        fig.set_size_inches(w=12, h=6)
        xdata = self.memory['time_stamp'][clip:]
        colors = ['blue','green','m','yellow','darkorange','r','r','r','r']

        # Get MJT Optimization -----------------------------
        N = 50 # N = len(xdata)
        xMJT = np.linspace(0,max(xdata),N)
        x0 = [self._start_pos,0,0]
        xf = [self._desired_pos,0,0]
        duration = [min(xdata), max(xdata)]

        # duration = [0,self.xdur_human]
        # xf = [self.goal_state[key] for key in ['pos', 'vel', 'acc']]

        optimizer = MinJerkOptimization(x0,xf,duration, N)
        J = optimizer.get_jerk()
        xt = optimizer.get_state_trajectory(J)

        # xt = self.H_MJTdes
        # xMJT = np.linspace(0,self.xdur_human,self.N_human)
        # J = self.H_MJT_J


        # optimizer.preview()



        MJT = {}
        MJT['pos'] = xt[:,0].flatten()
        MJT['vel'] = xt[:,1].flatten()
        MJT['acc'] = xt[:,2].flatten()
        MJT['jerk'] = J


        COL = 0 # ----------------------------------------------------------------------------
        key,r,c = 'pos',0,COL
        axs[r,c].plot(xdata, self.memory[key][clip:], label = '$x_{obs}$',ls='-')#,c=colors[iplt])
        axs[r,c].plot(xMJT, MJT[key], label = '$x_{MJT}$',ls=':')#,c=colors[iplt])
        axs[r,c].hlines(self.goal_state[key],xmin=min(xdata),xmax=max(xdata), color='k',ls=':',label = '$x_{goal}$')
        axs[r,c].set_ylabel(key)
        axs[r,c].legend()

        key,r,c = 'vel',1,COL
        axs[r,c].plot(xdata, self.memory[key][clip:], label = '$\dot{x}_{obs}$',ls='-')#, c=colors[iplt])
        axs[r,c].plot(xMJT, MJT[key], label = '$\dot{x}_{MJT}$',ls=':')#, c=colors[iplt])
        axs[r, c].hlines(self.goal_state[key], xmin=min(xdata), xmax=max(xdata), color='k', ls=':', label='$\dot{x}_{goal}$')
        axs[r,c].set_ylabel(key)
        axs[r,c].legend()

        key,r,c = 'acc',2,COL
        axs[r,c].plot(xdata, self.memory[key][clip:], label = '$\ddot{x}_{obs}$',ls='-')#, c=colors[iplt])
        axs[r,c].plot(xMJT, MJT[key], label = '$\ddot{x}_{MJT}$',ls=':')#, c=colors[iplt])
        axs[r, c].hlines(self.goal_state[key], xmin=min(xdata), xmax=max(xdata), color='k', ls=':', label='$\ddot{x}_{goal}$')
        axs[r,c].set_ylabel(key)
        axs[r,c].legend()

        key,r,c = 'jerk',3,COL
        axs[r,c].plot(xdata, self.memory[key][clip:], label='$\dddot{x}_{obs}$', ls='-')#, c=colors[iplt])
        axs[r,c].plot(xMJT[1:], MJT[key], label='$\dddot{x}_{MJT}$', ls=':')#, c=colors[iplt])
        axs[r,c].set_ylabel(key)
        axs[r,c].legend()

        COL = 1 # ----------------------------------------------------------------------------
        key, r, c = 'Fext', 0, COL
        axs[r, c].plot(xdata, self.memory[key][clip:], label='$F_{obs}$', ls='-')#, c=colors[iplt])
        axs[r, c].set_ylabel(key)
        axs[r, c].legend()

        key, r, c = 'dFextdt', 1, COL
        axs[r, c].plot(xdata, self.memory[key][clip:], label='$\dot{F}_{obs}$', ls='-')#, c=colors[iplt])
        axs[r, c].set_ylabel(key)
        axs[r, c].legend()

        key,r,c = 'action',2,COL
        axs[r,c].plot(xdata, self.memory[key][clip:], label='$K_{d}$', ls='-')#, c=colors[iplt])
        axs[r,c].set_ylabel(key)
        axs[r,c].legend()

        if epi_rewards is not None:
            key, r, c = 'epi_reward', 3, COL
            axs[r, c].plot(epi_rewards, label='$K_{d}$', ls='-')  # , c=colors[iplt])
            axs[r, c].set_ylabel(key)
            axs[r, c].legend()

        plt.show()
    def close(self):
        pass


    ####################################################################
    ## Getter Functions ################################################
    ####################################################################

    def get_human_force(self):

        """ get simulated human interaction forces """
        posd = self.goal_state['pos'] # desired position
        pos = self.full_state['pos'] # current position
        vel = self.full_state['vel'] # current velocity
        Kp = self.human['stiffness'] # human stiffness
        Kd = self.human['damping'] # human damping
        Fext = Kp*(posd - pos) - Kd*(vel)
        return Fext

    def get_robot_force(self,damping,stiffness = 0):
        """ get robot interaction forces based on chosen action """
        posd = self.goal_state['pos']  # desired position
        pos = self.full_state['pos']  # current position
        vel = self.full_state['vel']  # current velocity
        Kp,Kd = stiffness,damping
        Fcpl = Kp*(posd - pos) - Kd*(vel)
        return Fcpl



    ####################################################################
    ## Util Functions ##################################################
    ####################################################################
    def memory_dump(self):
        """ format memory as array and clear current memory buffer """
        mem_tmp = [np.copy(self.memory[key]) for key in self.state_names]
        self.memory_clear()
        return  np.array(mem_tmp)
    def memory_update(self):
        """ add current full state to memory buffer """
        for key in self.full_state:
            self.memory[key].append(self.full_state[key])
    def memory_clear(self):
        """ reset memory to empty """
        self.memory = {}
        for key in self.full_state:
            self.memory[key] = [self.full_state[key]]


    def check_done(self):
        # unpack
        error_thresh = self.terminal_conditions['error_thresh']
        max_duration = self.terminal_conditions['max_duration']

        # calc terminal conditions
        state_curr = np.array([self.full_state[key] for key in self.state_names])  # current state
        state_final = np.array([self.goal_state[key] for key in self.state_names])  # final/goal state
        state_final[0] = self._desired_pos
        is_final_state = np.all(np.abs(state_final - state_curr) < error_thresh)  # error from goal state
        exceeds_time_limit = (self.full_state['time_stamp'] > max_duration)  # episode time limite exceeded

        # return check (bool)
        if exceeds_time_limit or is_final_state:  done = True
        else: done = False
        return done

    def make_observation(self):
        return [self.full_state[key] for key in self.state_names]
# import math
#
# import numpy as np
#
# import HC
#
#
# class Environment(object):
#     action_set = []
#     state = []
#     #dt=0;
#
#     def __init__(self):
#         # cart position in the horizontal direction x = state[0], and cart velocity denoted by x_ = state[1]
#         # vertical angle of the pole denoted by tetha = state[2], angular velocity of the pole tetha_ = state[3]
#         self.x=0
#         self.x_list=[]
#         self.Time_stamps=0;
#         self.state = [0,0,0,0]
#         self.action_set = [5,22.5,50]
#         self.x=0
#         self.N = 100          # number of uniform timesteps to sample in optimization (decrease for computation speed)
#         self.x0 = [0,0,0]    # start [pos,vel,acc]
#         self.xf = [10,0,0]   # final [pos,vel,acc]
#         self.dur = [0,5]     # [t0,tf] start and end time of trajectory
#         self.optimizer = HC.MinJerkOptimization(self.x0,self.xf,self.dur,self.N)
#         self.optimizer.bias['myopia'], self.optimizer.bias['rushed'] = 0.0, 0.0 # default
#         self.J = self.optimizer.get_jerk()
#         self.xd = self.optimizer.get_state_trajectory(self.J)
#         self.dt=self.optimizer.dt
#         self.tt=0
#         self.r=0
#         self.JJ=0
#
#     def apply_action(self,action,phi):
#         j=1
#         reward=0
#         # while j<10:
#         #     u = phi*self.action_set[action]
#         #     self.get_current_state(u)
#         #     reward =reward-self.get_reward()
#         #     j=j+1
#
#         rewards = np.zeros(9)
#         for j in range(9):
#             u = phi * self.action_set[action]
#             self.get_current_state(u)
#             rewards[j] = - self.get_reward()
#         reward =  np.linalg.norm(rewards)
#         # reward = np.mean(rewards)
#         return reward, self.state
#
#     def get_state_variable(self,variable_name):
#
#         if variable_name == 'V':
#             return self.state[0]
#         elif variable_name == "F":
#             return self.state[1]
#         elif variable_name == "V_":
#             return self.state[2]
#         else:
#             return self.state[3]
#
#     def set_state_variable(self,variable_name,value):
#
#         if variable_name == 'V':
#             self.state[0] = value
#         elif variable_name == "F":
#             self.state[1] = value
#         elif variable_name == "V_":
#             self.state[2] = value
#         elif variable_name == "F_":
#             self.state[3] = value
#
#     def get_current_state(self,u):
#         M=1
#         V__old=self.get_state_variable('V_')
#         self.set_state_variable('V_',(1/M)*(self.get_state_variable('F')-u*self.get_state_variable('V')))
#         V__new=self.get_state_variable('V_')
#         self.JJ=(V__new-V__old)/self.dt
#         self.set_state_variable('V', self.get_state_variable('V') + (self.get_state_variable('V_') *self.dt))
#         self.x=self.x+self.get_state_variable('V')*self.dt;
#         self.x_list.append(self.x)
#         stiffness = 4.0 # increase for more aggressive/accurate tracking
#         damping = 1   # increase if unstable
#         mass = 0        # !!! need to double-check calculations for this !!!!
#         Hcontrol = HC.ImpedanceController(stiffness,damping,mass,self.xd[self.Time_stamps],self.dt)
#         Hcontrol.tauf = 3
#         F_old=self.get_state_variable('F')
#         self.set_state_variable('F',Hcontrol.get_force([self.x,self.get_state_variable('V'),self.get_state_variable('V_')],self.xd[self.Time_stamps]))
#         F_new=self.get_state_variable('F')
#         self.set_state_variable('F_',(F_new-F_old)/(self.dt))
#         self.Time_stamps=self.Time_stamps+1;
#     def get_reward(self):
#             r=self.JJ#*self.JJ
#             return r