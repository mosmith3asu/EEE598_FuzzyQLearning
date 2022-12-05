import copy

import numpy as np
import matplotlib.pyplot as plt
# from modules.HC import MinJerkOptimization
from modules.MJT import get_MJT


from Controller_Config import ALL_SETTINGS
PUBLIC = ALL_SETTINGS.public


class Environment(object):
    plt.ion()
    clip = 3
    nRows, nCols = 6, 2
    fig, axs = plt.subplots(nRows, nCols)
    fig.set_size_inches(w=28, h=12)

    _t = np.linspace(0, 1, 10)
    _y = np.linspace(0, 1, 10)

    ax_lines = np.empty([nRows, nCols, 3], dtype=object)
    ax_loc_dict = {}

    ################################################################
    COL = 0  #######################################################
    key, r, c = 'pos', 0, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$x_{obs}$', ls='-')[0])
    ax_lines[r, c, 1] = (axs[r, c].plot(_t, _y, label='$x_{MJT}$', ls=':')[0])
    ax_lines[r, c, 2] = (axs[r, c].plot(_t, np.ones(len(_t)), color='k', ls=':', label='$x_{goal}$')[0])
    axs[r, c].set_ylabel(key)
    axs[r, c].legend()

    key, r, c = 'vel', 1, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$\dot{x}_{obs}$', ls='-')[0])
    ax_lines[r, c, 1] = (axs[r, c].plot(_t, _y, label='$\dot{x}_{MJT}$', ls=':')[0])
    ax_lines[r, c, 2] = (axs[r, c].plot(_t, np.ones(len(_t)), color='k', ls=':', label='$\dot{x}_{goal}$')[0])
    axs[r, c].set_ylabel(key)
    axs[r, c].legend()

    key, r, c = 'acc', 2, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$\ddot{x}_{obs}$', ls='-')[0])
    ax_lines[r, c, 1] = (axs[r, c].plot(_t, _y, label='$\ddot{x}_{MJT}$', ls=':')[0])
    ax_lines[r, c, 2] = (axs[r, c].plot(_t, np.ones(len(_t)), color='k', ls=':', label='$\ddot{x}_{goal}$')[0])
    axs[r, c].set_ylabel(key)
    axs[r, c].legend()

    key, r, c = 'jerk', 3, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$\dddot{x}_{obs}$', ls='-')[0])
    ax_lines[r, c, 1] = (axs[r, c].plot(_t, _y, label='$\dddot{x}_{MJT}$', ls=':')[0])
    axs[r, c].set_ylabel(key)
    axs[r, c].legend()

    key, r, c = 'F', 4, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$F_{obs}$', ls='-')[0])
    axs[r, c].set_ylabel(key)
    axs[r, c].legend()

    key, r, c = 'dFdt', 5, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$\dot{F}_{obs}$', ls='-')[0])
    axs[r, c].set_ylabel(key)
    axs[r, c].legend()

    ################################################################
    COL = 1  #######################################################

    key, r, c = 'action', 0, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$K_{d}$', ls='-')[0])
    axs[r, c].set_ylabel(key)
    axs[r, c].legend()

    key, r, c = 'dt', 1, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$dt$', ls='-')[0])
    axs[r, c].set_ylabel(key)
    axs[r, c].legend()

    key, r, c = 'epi_jerk', 2, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$J_{epi}$', ls='-')[0])
    axs[r, c].set_ylabel('Jerk [$m/s^{3}$]')
    axs[r, c].legend()

    key, r, c = 'epi_energy', 3, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$E_{epi}$', ls='-')[0])
    axs[r, c].set_ylabel('Energy [joules]')
    axs[r, c].legend()

    key, r, c = 'epi_length', 4, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$E_{epi}$', ls='-')[0])
    axs[r, c].set_ylabel('Time [$s$]')
    axs[r, c].legend()

    key, r, c = 'epi_reward', 5, COL
    ax_loc_dict[key] = [r, c]
    ax_lines[r, c, 0] = (axs[r, c].plot(_t, _y, label='$R_{epi}$', ls='-')[0])
    axs[r, c].set_ylabel('Cum. Reward')
    axs[r, c].legend()

    def __init__(self,
                 desired_pos, timestep,
                 #error_thresh, max_duration,
                 start_pos,
                 **kwargs
                 # desired_pos, timestep,
                 # mass = 1.3, K_human=1.0, D_human = 0.1,
                 # error_thresh=0.1,max_duration=10., start_pos = 0.,**kwargs
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
        self.f_pos2dist = PUBLIC['f_pos2dist']

        self.expected_timestep = timestep
        self.state_names = ['vel', 'acc', 'F', 'dFdt']
        self._desired_pos = self.f_pos2dist(desired_pos)
        self._start_pos = self.f_pos2dist(start_pos)


        # Initialized Aux Variables (need to call reset() )
        self.goal_state = None
        self.full_state = None
        self.memory = None

        self.started_moving = False
        self.start_time = 0

    ####################################################################
    ## Core Functions ##################################################
    ####################################################################
    def step(self, state, action):

        if self.started_moving == False:
            if state['F'] > 0.5: self.started_moving = True
            else: self.start_time = state['timestamp']


        if self.started_moving:
            pos0 = self.full_state['pos']
            # vel0    = self.full_state['vel']
            # acc0    = self.full_state['acc']
            # F0      = self.full_state['F']
            # dFdt    = self.full_state['dFdt']
            # jerk0   = self.full_state['jerk']
            # tsec0   = self.full_state['timestamp']
            # dtsec0  = self.full_state['dt']
            energy0 = self.full_state['energy']

            pos1 = self.f_pos2dist(state['pos']) - self._start_pos
            # vel1 = self.f_pos2dist(state['vel'])
            # acc1 = self.f_pos2dist(state['acc'])
            # jerk1 = self.f_pos2dist(state['jerk'])
            vel1 = state['vel']
            acc1 = state['acc']
            jerk1 = state['jerk']
            F1 = state['F']
            dFd1 = state['dFdt']

            tsec1 = state['timestamp'] - self.start_time
            dtsec1 = state['dt']
            energy1 = energy0 + (abs(F1) * (pos1 - pos0))

            self.full_state['pos'] = pos1
            self.full_state['vel'] = vel1
            self.full_state['acc'] = acc1
            self.full_state['F'] = F1
            self.full_state['dFdt'] = dFd1
            self.full_state['jerk'] = jerk1
            self.full_state['timestamp'] = tsec1
            self.full_state['dt'] = dtsec1
            self.full_state['energy'] = energy1
            self.full_state['action'] = action

            reward = jerk1
            obs = self.make_observation()
            info = 'started'
        else:
            reward = None
            obs = None
            info = 'waiting'
            self.start_time = state['timestamp']
        return obs, reward, info

    def reset(self):
        self.goal_state = {}
        self.goal_state['pos'] =self._desired_pos
        self.goal_state['vel'] = 0.
        self.goal_state['acc'] = 0.
        self.goal_state['F'] = 0.
        self.goal_state['dFdt'] = 0.
        self.goal_state['jerk'] = 0.

        self.full_state = {}
        self.full_state['pos'] =self._start_pos
        self.full_state['vel'] = 0.
        self.full_state['acc'] = 0.
        self.full_state['F'] = 0.
        self.full_state['dFdt'] = 0.
        self.full_state['jerk'] = 0.
        self.full_state['timestamp'] = 0.
        self.full_state['action'] = 0
        self.full_state['energy'] = 0
        self.full_state['dt'] = 0.

        self.started_moving = False
        #self.start_time = 0

        self.memory_clear()
        self.memory_update()
        obs = self.make_observation()
        return obs

    def _update_line_data(self, r, c, idata, ydata, timestamps=None):
        xdata = np.linspace(0, 1, len(ydata)) if timestamps is None else timestamps
        self.ax_lines[r, c, idata].set_ydata(ydata)
        self.ax_lines[r, c, idata].set_xdata(xdata)
        self.ax_lines[r, c, idata].axes.relim()
        self.ax_lines[r, c, idata].axes.autoscale_view()

    def render(self, epi_reward=None, epi_energy=None, epi_length=None, epi_jerk=None, show=True):

        #xdata = self.memory['timestamp'][self.clip:]

        try:
            N = 50
            _t = np.mean(np.array(self.memory['timestamp'][1:])-np.array(self.memory['timestamp'][:-1]))
            _v = np.gradient(self.memory['pos'],_t).round(2)
            _a = np.gradient(_v,_t).round(2)
            _J = np.gradient(_a,_t).round(2)
            print(f"RENDER: t[{[min(self.memory['timestamp']),max(self.memory['timestamp'])]}] => {_t}")
            print(f"RENDER: p{[min(self.memory['pos']), max(self.memory['pos'])]}]")
            print(f"RENDER: v{[min(self.memory['vel']), max(self.memory['vel'])]}] => {[min(_v), max(_v)]}")
            print(f"RENDER: a[{[min(self.memory['acc']), max(self.memory['acc'])]}] => {[min(_a), max(_a)]}")
            print(f"RENDER: j[{[min(self.memory['jerk']), max(self.memory['jerk'])]}] => {[min(_J), max(_J)]}")

            p0 = min(self.memory['pos'][self.clip:]) # self._start_pos
            pf = max(self.memory['pos'][self.clip:]) # self._desired_pos
            tf = max(self.memory['timestamp'][self.clip:])
            t0 = min(self.memory['timestamp'][self.clip:])
            xt = np.array(get_MJT(tf=tf, xf=pf, t0=t0, x0=p0, N=N))

            # Get MJT Optimization -----------------------------
            MJT = {}
            MJT['t'] = np.linspace(min(self.memory['timestamp']), max(self.memory['timestamp']), N)
            MJT['pos']  = xt[0, :].flatten()
            MJT['vel']  = np.gradient(xt[0,:],_t) #xt[1, :].flatten()#*self.expected_timestep
            MJT['acc']  = np.gradient(xt[1,:],_t) #xt[2, :].flatten()#/self.expected_timestep
            MJT['jerk'] = np.gradient(xt[2,:],_t) #xt[3, :].flatten()#/pow(self.expected_timestep,2)


            T_obs = self.memory['timestamp'][self.clip:]
            T_MJT = np.linspace(min(T_obs), max(T_obs), N)
            T_norm = np.linspace(0, 1, N)

            iobs = 0
            iMJT = 1
            iGoal = 2

            # for key in ['jerk']: # 'pos','vel','acc'
            #     self.memory[key] = self.memory[key][self.clip:]

            key = 'pos'
            r, c = self.ax_loc_dict[key]
            ydata = self.memory[key][self.clip:]
            self._update_line_data(r, c, iobs, ydata, timestamps=T_obs)
            self._update_line_data(r, c, iMJT, ydata=MJT[key], timestamps=T_MJT)
            self._update_line_data(r, c, iGoal, ydata=self.goal_state[key] * np.ones(len(T_obs)),
                                   timestamps=T_obs)

            key = 'vel'
            r, c = self.ax_loc_dict[key]
            ydata = self.memory[key][self.clip:]
            # ydata = np.gradient(self.memory['pos'][self.clip:]) / self.expected_timestep
            self._update_line_data(r, c, iobs, ydata[1:], timestamps=T_obs[1:])
            self._update_line_data(r, c, iMJT, ydata=MJT[key], timestamps=T_MJT)
            self._update_line_data(r, c, iGoal, ydata=self.goal_state[key] * np.ones(len(T_obs)),
                                   timestamps=T_obs)

            key = 'acc'
            r, c = self.ax_loc_dict[key]
            ydata = self.memory[key][self.clip:]
            # ydata = np.gradient(self.memory['vel'][self.clip:]) / self.expected_timestep
            self._update_line_data(r, c, iobs, ydata, timestamps=T_obs)
            self._update_line_data(r, c, iMJT, ydata=MJT[key], timestamps=T_MJT)
            self._update_line_data(r, c, iGoal, ydata=self.goal_state[key] * np.ones(len(T_obs)),
                                   timestamps=T_obs)

            key = 'jerk'
            r, c = self.ax_loc_dict[key]
            ydata = self.memory[key][self.clip:]
            # ydata = np.gradient(self.memory['acc'][self.clip:]) / self.expected_timestep
            self._update_line_data(r, c, iobs, ydata, timestamps=T_obs)
            self._update_line_data(r, c, iMJT, ydata=MJT[key], timestamps=T_MJT)

            key = 'F'
            r, c = self.ax_loc_dict[key]
            self._update_line_data(r, c, iobs, ydata=self.memory[key][self.clip:], timestamps=T_obs)

            key = 'dFdt'
            r, c = self.ax_loc_dict[key]
            self._update_line_data(r, c, iobs, ydata=self.memory[key][self.clip:], timestamps=T_obs)

            key = 'action'
            r, c = self.ax_loc_dict[key]
            self._update_line_data(r, c, iobs, ydata=self.memory[key][self.clip:], timestamps=T_obs)

            key = 'dt'
            r, c = self.ax_loc_dict[key]
            self._update_line_data(r, c, iobs, ydata=self.memory[key][self.clip:], timestamps=T_obs)

            if epi_energy is not None:
                key = 'epi_energy'
                r, c = self.ax_loc_dict[key]
                self._update_line_data(r, c, iobs, ydata=epi_energy, timestamps=np.arange(0, len(epi_energy)))
            if epi_reward is not None:
                key = 'epi_reward'
                r, c = self.ax_loc_dict[key]
                self._update_line_data(r, c, iobs, ydata=epi_reward, timestamps=np.arange(0, len(epi_reward)))
            if epi_length is not None:
                key = 'epi_length'
                r, c = self.ax_loc_dict[key]
                self._update_line_data(r, c, iobs, ydata=epi_length, timestamps=np.arange(0, len(epi_length)))
            if epi_jerk is not None:
                key = 'epi_jerk'
                r, c = self.ax_loc_dict[key]
                self._update_line_data(r, c, iobs, ydata=epi_jerk, timestamps=np.arange(0, len(epi_jerk)))

            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

            if show: plt.show()
        except Exception as e:
            print(f'ENV RENDER ERROR \n {e}')

    def close(self):
        pass

    ####################################################################
    ## Getter Functions ################################################
    ####################################################################

    def get_robot_force(self, damping, stiffness=0):
        """ get robot interaction forces based on chosen action """
        posd = self.goal_state['pos']  # desired position
        pos = self.full_state['pos']  # current position
        vel = self.full_state['vel']  # current velocity
        Kp, Kd = stiffness, damping
        Fcpl = Kp * (posd - pos) - Kd * (vel)
        return Fcpl

    ####################################################################
    ## Util Functions ##################################################
    ####################################################################
    def memory_dump(self):
        """ format memory as array and clear current memory buffer """
        mem_tmp = [np.copy(self.memory[key]) for key in self.state_names]
        self.memory_clear()
        return np.array(mem_tmp)

    def memory_update(self):
        """ add current full state to memory buffer """
        for key in self.full_state:
            self.memory[key].append(self.full_state[key])

    def memory_clear(self):
        """ reset memory to empty """
        self.memory = {}
        for key in self.full_state:
            self.memory[key] = [self.full_state[key]]

    def memory_save(self,path, **kwargs):
        save_dict = copy.deepcopy(self.memory)
        for key in kwargs.keys():
            save_dict[key] = kwargs[key]
        print('saving...')
        print(f'{save_dict.keys()}')
        #np.save(path, **save_dict)
        np.savez_compressed(path, a=save_dict)

    def make_observation(self):
        return [self.full_state[key] for key in self.state_names]
