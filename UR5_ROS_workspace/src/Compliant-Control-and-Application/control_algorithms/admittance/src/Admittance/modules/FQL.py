#!/usr/bin/env python3
import numpy as np

try:
    from modules.FIS import FuzzyInferenceSystem
except:
    from FIS import FuzzyInferenceSystem

from Controller_Config import ALL_SETTINGS

SETTINGS = ALL_SETTINGS.learning_node

FIS_settings = SETTINGS['FIS_settings']
FIS = FuzzyInferenceSystem()
FIS.add_fuzzy_set('v', FIS_settings['v_range'], FIS_settings['v_bins'])
FIS.add_fuzzy_set('a', FIS_settings['dvdt_range'], FIS_settings['dvdt_bins'])
FIS.add_fuzzy_set('F', FIS_settings['F_range'], FIS_settings['F_bins'])
FIS.add_fuzzy_set('dFdt', FIS_settings['dFdt_range'], FIS_settings['dFdt_bins'])
FIS.generate_rules()


class FQLhandler(object):
    def __init__(self, alpha, gamma, lam, sp, theta):

        self.ax_action = 1
        self.Si = np.arange(FIS.nRules)  # list of all rule indicies
        self.A = SETTINGS['crisp_actions']   # discrete crisp actions
        self.Ai = self.A * np.ones([len(self.Si), len(self.A)])  # discrete crisp actions for every state
        self.num_actions = len(self.A)
        self.num_states = len(self.Si)

        self.q = np.zeros([len(self.Si), len(self.A)])

        self.alpha = alpha
        self.gamma = gamma  # γ must be selected high enough so that the agent will try to collect long term rewards during an episode

        self.lam = lam

        self.eligibility_trace = np.zeros([self.num_states, self.num_actions])

        self.action_frequency = np.zeros([self.num_states, self.num_actions])
        self.w_dir_explore = theta
        self.w_undir_explore = sp

    def eval_Ut(self, Ut):
        """Ut is given as [ai,phi_i] and this evalutes to continous global action"""
        ai, phi_i = Ut
        checksum = np.sum(phi_i)
        if abs(1 - checksum) > 1e-4:
            raise Exception(f'phi Checksum error = {np.sum(phi_i)}')

        # Ut = np.sum(self.Ai[:,ai] * phi_i)
        Ut = 0
        for i in range(self.num_states):
            Ut += self.A[ai[i]] * phi_i[i]
        return Ut

    def get_Ut_Xt(self, Xt):
        """See [Fuzzy inference system learning by reinforcement methods]"""
        # get rule strength
        phi_i = FIS.make_inference(Xt)  # FIS estimate rule strength

        # Undirected Exploration
        # sp = 0.5  # noise size w.r.t. the range of qualities (decrease => less exploration)
        sp = self.w_undir_explore
        psi = np.random.exponential(
            size=[self.num_states, self.num_actions])  # exponential dist. scaled to match range of q-values
        qi_range = np.max(self.q, axis=self.ax_action) - np.min(self.q, axis=self.ax_action)
        iuniform = np.array(np.where(qi_range > 1e3)).flatten()
        sf = (sp * qi_range / np.max(psi)).reshape([self.num_states, -1])  # corrasponding scaling factor
        sf[iuniform] = 1
        eta = sf * psi

        # Directed exploration
        #   The directed term gives a bonus to the actions that have
        #   been rarely elected
        theta = self.w_dir_explore  # positive factor used to weight the directed exploration
        nt_XtUt = self.action_frequency  # the number of time steps in which action U has been elected
        rho = theta / np.exp(nt_XtUt)

        # Choose action
        ai = np.argmax(self.q + eta + rho, axis=self.ax_action)
        self.action_frequency[:, ai] = (1) * phi_i  # update selected action freq
        return ai, phi_i

    def get_Qt_XtUt(self, Xt, Ut):
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
            Qt_UtXt = np.sum(self.q[self.Si, ai] * phi_i)
            return Qt_UtXt

    def update_Q(self, et0, Xt0, Ut0, Xt1, rt1):
        """
        Updated each iteration of the algorithm
        :param et0: eligibility trace
        :param Xt0: state before new observation
        :param Ut0: action taken before new observation
        :param Xt1: new observation
        :param rt1: reward of new observation
        :return:
        """
        Q_XtUt0 = self.get_Qt_XtUt(Xt0, Ut=Ut0)
        Qstar_Xt1 = self.get_Qt_XtUt(Xt1, Ut='optimal')

        # for a rule is given by the Q * -function:
        td_error = rt1 + self.gamma * Qstar_Xt1 - Q_XtUt0
        self.q = self.q + self.alpha * td_error * et0

    def update_EligibiltyTrace(self, Ut):
        ai, phi_i = Ut
        et = self.eligibility_trace
        et = (self.gamma * self.lam) * et  # decay trace
        et[self.Si, ai] += phi_i  # add strength to current action eligibility
        self.eligibility_trace = et
        return self.eligibility_trace

    def memory2reward(self, jerk_samples, ord=2):
        """
        :param jerk_samples: tau2t jerk samples from faster admittance controller
        :param tau_f: duration of motion in discrete steps (unkown before hand)
        :param N: non-negative cumulative jerk over duration of discrete motion tau_f
        :return:
        """
        #jerk_samples = [J*int(abs(J) > 0.5) for J in jerk_samples]
        N = -1 * np.power(np.linalg.norm(jerk_samples,ord=ord), 3)
        return N

    def get_action(self, Xt):
        ai, phi_i = self.get_Ut_Xt(Xt)
        return [ai, phi_i]
