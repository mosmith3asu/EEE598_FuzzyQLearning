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

import numpy as np
from dataclasses import dataclass
from Simulation.modules.FIS import FuzzyInferenceSystem
from Simulation.modules.Environment import Environment
import matplotlib.pyplot as plt



def obj2dict(obj):
    # self = super().self
    obj_dict = {}
    for key in obj.__dict__.keys():
        if not key.startswith('_'):
            obj_dict[key] = obj.__dict__[key]
    return obj_dict

@dataclass
class FQL_Config:
    num_episodes: int = 20
    max_epi_duration: int = 15
    fs_admittance: int = 100
    fs_learning: int = 10
    fs_k: int = fs_admittance/fs_learning

    FIS: object = type('config_obj', (object,), {})
    FIS.v_range, FIS.v_bins = [-2, 4], 4
    FIS.F_range, FIS.F_bins = [-8, 8], 4
    FIS.dvdt_range, FIS.dvdt_bins = [-5, 5], 4
    FIS.dFdt_range, FIS.dFdt_bins = [-6, 6], 4


    FQL: object = type('config_obj', (object,), {})
    FQL.alpha = 0.05
    FQL.gamma = 0.8
    FQL.lamb = 0.5 # eligibility trace
    FQL.sp = 1 # Undirected Exploration noise
    FQL.theta = 10 # direct exploration scale
    FQL.crisp_actions = [10, 22.5, 50]

    ENV: object = type('config_obj', (object,), {})
    ENV.desired_pos = 5.
    ENV.timestep = 1/fs_admittance
    ENV.mass= 1.3
    ENV.K_human= 10.0
    ENV.D_human= 2.0
    ENV.rush_human = -0.5
    ENV.xdur_human = 5.0
    ENV.N_human = 100
    ENV.error_thresh = 0.01
    ENV.max_duration= max_epi_duration
    ENV.start_pos= 0.
    ENV.timeseries =  np.arange(0, ENV.max_duration, ENV.timestep)



FIS = FuzzyInferenceSystem()
FIS.add_fuzzy_set('v', [-10, 10], 3)
FIS.add_fuzzy_set('a', [-5, 5], 3)
FIS.add_fuzzy_set('F', [-5, 5], 3)
FIS.add_fuzzy_set('dFdt', [-5, 5], 3)
FIS.generate_rules()


class FQLhandler(object):
    def __init__(self,alpha=0.05,gamma=0.8,epsilon = 0.7):
        self.q = np.zeros([5,5])
        self.ax_action = 1
        self.Si = np.arange(FIS.nRules) # list of all rule indicies
        self.A =   [10, 22.5, 50]# discrete crisp actions
        self.Ai = self.A*np.ones([len(self.Si), len(self.A)])  # discrete crisp actions for every state
        self.epsilon = epsilon
        self.num_actions = len(self.A)
        self.num_states = len(self.Si)

        self.q = np.zeros([len(self.Si),len(self.A)])

        self.alpha = alpha
        self.gamma = gamma # γ must be selected high enough so that the agent will try to collect long term rewards during an episode

        self.eligibility_trace = np.zeros([self.num_states,self.num_actions])
        self.lam = 0.5

        self.action_frequency = np.zeros([self.num_states, self.num_actions])
        self.w_dir_explore = 10
        self.w_undir_explore = 1

    def eval_Ut(self,Ut):
        """Ut is given as [ai,phi_i] and this evalutes to continous global action"""
        ai, phi_i = Ut
        checksum = np.sum(phi_i)
        if abs(1-checksum) > 1e-4:
            raise Exception(f'phi Checksum error = {np.sum(phi_i)}')

        # Ut = np.sum(self.Ai[:,ai] * phi_i)
        Ut = 0
        for i in range(self.num_states):
            Ut += self.A[ai[i]]*phi_i[i]
        return Ut

    def get_Ut_Xt_EPSILON_GREEDY(self,Xt):
    # def get_Ut_Xt(self, Xt):
        epsilon = self.epsilon
        phi_i = FIS.make_inference(Xt) # FIS estimate rule strength
        ai = np.empty(self.num_states,dtype=int)
        for i in range(self.num_states):
            if np.random.rand(1) < epsilon: # exploint
                ai[i] = np.argmax(self.q[i,:])
            else: # explore
                ai[i] = np.random.choice(np.arange(self.num_actions))
        return ai,phi_i

    # def get_Ut_Xt_DIRECTED(self, Xt):
    def get_Ut_Xt(self, Xt):
        """See [Fuzzy inference system learning by reinforcement methods]"""
        # get rule strength
        phi_i = FIS.make_inference(Xt)  # FIS estimate rule strength


        # Undirected Exploration
        # sp = 0.5  # noise size w.r.t. the range of qualities (decrease => less exploration)
        sp =  self.w_undir_explore
        psi = np.random.exponential(size=[self.num_states,self.num_actions])  # exponential dist. scaled to match range of q-values
        qi_range = np.max(self.q, axis=self.ax_action) - np.min(self.q, axis=self.ax_action)
        iuniform = np.array(np.where(qi_range > 1e3)).flatten()
        sf = (sp*qi_range/np.max(psi)).reshape([self.num_states,-1])  # corrasponding scaling factor
        sf[iuniform] = 1
        eta = sf*psi

        # Directed exploration
        #   The directed term gives a bonus to the actions that have
        #   been rarely elected
        theta = self.w_dir_explore  # positive factor used to weight the directed exploration
        nt_XtUt = self.action_frequency  # the number of time steps in which action U has been elected
        rho = theta / np.exp(nt_XtUt)

        # Choose action
        ai = np.argmax(self.q + eta + rho,axis=self.ax_action)
        self.action_frequency[:, ai] = (1) * phi_i # update selected action freq
        return ai,phi_i

    def get_Qt_XtUt(self,Xt,Ut):
        """
        :param Xt:
        :param Ut: continous global action written as
        :return:
        """
        phi_i = FIS.make_inference(Xt)  # FIS estimate rule strength
        if Ut == 'optimal':
            qi_astar = np.argmax(self.q, axis=self.ax_action)  # optimal quality for each rule
            Qtstar_Ut = np.sum(qi_astar * phi_i)
            return Qtstar_Ut
        else:
            ai, phi_i = Ut
            Qt_UtXt = np.sum(self.q[self.Si,ai] * phi_i)
            return Qt_UtXt


    def update_Q(self,et0,Xt0,Ut0,Xt1,rt1):
        """
        Updated each iteration of the algorithm
        :param et0: eligibility trace
        :param Xt0: state before new observation
        :param Ut0: action taken before new observation
        :param Xt1: new observation
        :param rt1: reward of new observation
        :return:
        """
        ai,phi_i = Ut0
        #qt0 = self.q
        #et0 = self.eligibility_trace
        Q_XtUt0 = self.get_Qt_XtUt(Xt0, Ut = Ut0)
        Qstar_Xt1 = self.get_Qt_XtUt(Xt1, Ut = 'optimal')


        # for a rule is given by the Q * -function:
        td_error = rt1 + self.gamma * Qstar_Xt1 - Q_XtUt0
        #self.q[:,ai] =  self.q[:,ai] + self.alpha * td_error * et0
        self.q = self.q + self.alpha * td_error * et0


    def update_EligibiltyTrace(self,Ut):
        ai,phi_i = Ut
        et = self.eligibility_trace
        et = (self.gamma*self.lam)*et # decay trace
        et[self.Si,ai] += phi_i       # add strength to current action eligibility
        self.eligibility_trace = et
        return self.eligibility_trace

    def memory2reward(self,jerk_samples):
        """
        :param jerk_samples: tau2t jerk samples from faster admittance controller
        :param tau_f: duration of motion in discrete steps (unkown before hand)
        :param N: non-negative cumulative jerk over duration of discrete motion tau_f
        :return:
        """
        jerk_samples = np.array(jerk_samples).flatten()
        N = -1*(np.power(np.linalg.norm(jerk_samples,ord=2),2))
        # N = -1*np.mean(np.abs(jerk_samples))
        return N

    def get_action(self,Xt):
        ai,phi_i = self.get_Ut_Xt(Xt)
        return [ai,phi_i]



def simulate_solo_human():
    # env = Environment(**env_params)

    env = Environment(**obj2dict(FQL_Config.ENV))

    env.reset()

    robot_damping = 0
    for itau, tau in enumerate(FQL_Config.ENV.timeseries):
        statej, reward, done, _ = env.step(action = robot_damping)
        env.memory_update()
        if done: break
    env.render()


def main():
    """
    Ri: rule i,...,n
    A = {a1,..,aj}: action set containing j descrete numbers of crisp action values
    Xt: continous global state
    Ut: continous global action output composed of actions selected on each rule
    Si=>Ri: fuzzy state of the rule Ri composed by a vector if fuzzy sets
    phi_i: firing strength of rule Ri
    q(Si,aj): q-value that determines the probability of choosing the action
    ai_prime: selected actions for each rule given policy

    :return:
    """
    nRows, nCols = 3, 1
    fig, axs = plt.subplots(nRows, nCols)

    for rush in [-0.9,0.9]:
        FQL_Config.ENV.rush_human = rush
        # simulate_solo_human()

        # SET UP PLOTS
        # plt.ion()
        # nRows, nCols = 4,1
        # lines = np.empty([nRows,nCols],dtype=object)
        # names = np.empty([nRows, nCols], dtype=object)
        # names[0, 0] = 'Epi Reward'
        # names[1, 0] = 'Epi Length'
        # names[2, 0] = 'V'
        # names[3, 0] = 'D'
        # fig,axs = plt.subplots(nRows,nCols)
        # axs = np.array(axs).reshape(nRows,nCols)
        # _x = np.arange(10)
        # _y = np.ones(10)
        #
        # for r in range(nRows):
        #     for c in range(nCols):
        #         lines[r,c], = axs[r,c].plot(_x,_y)
        #         axs[r, c].set_ylabel(names[r,c])

        # INTIALIZE TRAINING

        model = FQLhandler()
        env = Environment(**obj2dict(FQL_Config.ENV))
        time4update = lambda _itau: (_itau % FQL_Config.fs_k == 0)

        n_states = model.q.shape[0]
        n_actions = model.q.shape[1]
        # eligibility = np.zeros(n_states)
        #et = EligibilityTrace(n_states,gamma=model.gamma,lam=0.95)


        stats = {}
        stats['epi_reward'] = []
        stats['epi_len'] = []
        stats['epi_velocities'] = []
        stats['epi_dampings'] = []

        n_episodes = FQL_Config.num_episodes#FQL_settings['num_episodes']
        n_warmup = 5
        n_perform = 5

        min_epsilon = 0.1
        max_epsilon = 0.95
        episode_epsilon = np.linspace(min_epsilon,max_epsilon,n_episodes-(n_warmup+n_perform))
        episode_epsilon = np.hstack([ min_epsilon*np.ones(n_warmup),episode_epsilon, max_epsilon*np.ones(n_perform)])

        min_sp = 0.1
        max_sp = 0.9
        episode_sp = np.linspace(max_sp, min_sp, n_episodes - (n_warmup + n_perform))
        episode_sp = np.hstack([max_sp * np.ones(n_warmup), episode_sp, min_sp * np.ones(n_perform)])

        for ith_episode in range(n_episodes):
            model.epsilon = episode_epsilon[ith_episode]
            model.w_undir_explore = episode_sp[ith_episode]

            stats['epi_reward'].append(0)
            stats['epi_len'].append(0)
            stats['epi_velocities'] = []
            stats['epi_dampings'] = []

            Xt0 = env.reset()            # reset environment
            Ut0 = model.get_action(Xt0)  # select action from policy
            rt = []                     # reward buffer for intermediate rewards

            for itau, tau in enumerate(FQL_Config.ENV.timeseries):
                ai,phi_i = Ut0
                Cd = model.eval_Ut(Ut0)
                #statej, reward, done, _ = env.step(action=robot_damping)
                Xt1, _rt1, done, _ = env.step(Cd)
                rt.append(_rt1)



                # -------- update policy --------
                if time4update(itau):
                    # close previous learning update
                    rt1 = model.memory2reward(rt)
                    et0 = model.update_EligibiltyTrace(Ut0)
                    model.update_Q(et0, Xt0, Ut0, Xt1, rt1)
                    stats['epi_reward'][-1] += rt1
                    stats['epi_len'][-1] = env.full_state['time_stamp']

                    print(f'\r[[{rush}] e={ith_episode} itau={itau}] ',end='')
                    print(f'eps = {round(model.epsilon, 2)} ', end='')
                    print(f'rtau = {round(_rt1,2)} ',end='')
                    print(f'Cd = {round(Cd,2)} ', end='')
                    print(f'Q = {[round(np.min(model.q), 2),round(np.max(model.q), 2)]} ', end='')
                    #print(f'[V = {round(env.full_state["vel"],2)} ', end='')
                    print(f'')

                    # Open new learning step
                    rt = []  # reward buffer for intermediate rewards
                    Xt0 = Xt1  # update current state
                    Ut0 = model.get_action(Xt0)  # select action from policy

                    # xdata = np.arange(len(stats['epi_reward']))
                    # lines[0, 0].set_ydata(stats['epi_reward'])
                    # lines[0,0].set_xdata(xdata)
                    # fig.canvas.draw()
                    # fig.canvas.flush_events()


                # Close timestep
                    env.memory_update() #memory.add(reward, v, dvdt, F, dFdt)
                stats['epi_len'][-1] = tau
                #statei = np.copy(statej)
                if done: break

            # CLOSE EPISODE ===============================================
            # plt.close()
            # print(f'\t\t || Repi = {stats["epi_reward"][-1]}')
            # env.render(epi_rewards=stats['epi_reward'])
            # plt.show(blocking=False)
            # time.sleep(3)
            # plt.close()

        # Close FQL ========================================================

        # axs[0].set_title()
        skew = 'right' if  rush<0 else 'left'
        name = f'skew {skew}'
        axs[0].plot(stats['epi_reward'],label = name) # axs[1].plot(stats['epi_len'])
        axs[1].plot(env.memory['time_stamp'],env.memory['vel'],label = name)
        axs[2].plot(env.memory['time_stamp'],env.memory['action'],label = name)
        ylabels = ['Episode Reward','Velocity','Damping']
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        for i in range(nRows):
            axs[i].set_ylabel(ylabels[i])
       # env.render(epi_rewards=stats['epi_reward'])
    fig.savefig('Fig_SimulationResults2.png')
    plt.ioff()
    plt.show()





# Comu_reward=-1000000
# xtol = np.array([0.1,0.1,0.1])
# max_iter =int(2*env.N/10)
# nEpisodes = 50
#

# stats = {}
# stats['epi_reward'] = []
# stats['epi_len'] = []
#




if __name__=="__main__":
    main()



