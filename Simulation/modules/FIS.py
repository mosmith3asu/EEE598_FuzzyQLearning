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
import itertools
import matplotlib.pyplot as plt

class Triangles(object):
    def __init__(self, loc, size):
        self.loc = loc
        self.width = size
        self.amp = 1.0
        self.is_LB = False
        self.is_RB = False

    def membership(self, x):
        if self.is_LB and x < self.loc:
            membership = 1
        elif self.is_RB and x > self.loc:
            membership = 1
        else:
            membership = max(0, -1.0 / (self.width) * abs(x - self.loc) + self.amp)
        return membership


class FuzzySets(object):
    def __init__(self, xrange, nSets):
        self.nSets = nSets
        set_centers = np.linspace(xrange[0], xrange[1], nSets)
        set_size = (xrange[1] - xrange[0]) / (nSets - 1)
        self.fuzzy_sets = []
        for loc in set_centers: self.fuzzy_sets.append(Triangles(loc, set_size))
        self.fuzzy_sets[0].is_LB = True
        self.fuzzy_sets[-1].is_RB = True
        self.set_values = np.copy(set_centers)

    def memberships(self, val):
        memberships = [fset.membership(val) for fset in self.fuzzy_sets]
        return np.array(memberships)


class FuzzyInferenceSystem(object):
    def __init__(self):
        self.state_names = []
        self.state_vars = {}
        self.Rules = []
        self.Si = []
        self.nRules = 0
        self.nSets = 0

    def add_fuzzy_set(self, name, xrange, k):
        # self.state_names.append(name)
        # self.state_vars.append(StateVariable(xrange, N, set_size))
        self.state_vars[name] = FuzzySets(xrange, k)
        self.nSets += 1

    def generate_rules(self):
        self.Rules = itertools.product(*[self.state_vars[key].fuzzy_sets for key in self.state_vars])
        self.Rules = np.array(list(self.Rules))
        self.Si = np.array([[triangle.loc for triangle in rule] for rule in self.Rules])
        #self.RulePoints = np.array([[[triangle.loc-triangle.width,triangle.loc,triangle.loc+triangle.width] for triangle in rule] for rule in self.Rules])
        self.nRules = self.Rules.shape[0]

    def rule_strength(self, state):
        """phi_i firing strength of the rule Ri"""
        Ri_strength = np.zeros(self.nRules)
        for irule, rule in enumerate(self.Rules):
            phi = 1
            for istate in range(self.nSets):
                phi *= rule[istate].membership(state[istate])
            Ri_strength[irule] = phi
        return Ri_strength / np.sum(Ri_strength)

    def make_inference(self,Xt):
        """ Converts continous global action into fuzzy sets and strengths """
        phi_i = self.rule_strength(Xt)
        #Si = self.Si
        return phi_i



if __name__ == "__main__":
    k = 5
    FIS = FuzzyInferenceSystem()
    FIS.add_fuzzy_set('v', [-2.5, 2.5], k)
    FIS.add_fuzzy_set('a', [-5, 5], k)
    FIS.add_fuzzy_set('F', [-25, 25], k)
    FIS.add_fuzzy_set('dFdt', [-200, 200], k)
    FIS.generate_rules()

    state = [-5, -3, -2, 0]

    ranges = [[-2.5, 2.5], [-5, 5],[-25, 25], [-200, 200]]

    print(f'{FIS.nRules}')
    print(f'{FIS.Rules[0][0].membership(-5)}')
    print(f'{FIS.rule_strength(state)}')
    print(f'CHECKSUM: {np.sum(FIS.rule_strength(state))}')

    # RulePoints = FIS.RulePoints
    nRows,nCols = 4,1
    fig,axs = plt.subplots(nRows,nCols)
    fig.set_size_inches(w=8, h=10)
    axs[0].set_title('Fuzzy Inference System')
    # for rule
    conv_name = {'v': '$v$','a':'\dot{v}','F':'F_{H}','dFdt':'\dot{F}'   }
    for istate,state_name in enumerate(['v','a','F','dFdt']):

        rng = ranges[istate]
        rng = [-50,50]
        t = np.linspace(1.1*rng[0],1.1*rng[1],10000)
        fuzzy_set = FIS.state_vars[state_name]


        for iset in range(k):
            triangle = fuzzy_set.fuzzy_sets[iset]
            loc = triangle.loc
            w = triangle.width
            m = [triangle.membership(x) for x in t]

            lbls = [round(loc-w,1),round(loc+w,1)]
            if iset == 0: lbls[0] = -np.inf
            if iset == k-1: lbls[-1] = np.inf

            # name = conv_name[state_name]
            axs[istate].plot(t,m,label = '$\hat{'+f'{state_name}'+'}'+f'_{iset+1}'+'$ = '+f'{[min(lbls),max(lbls)]}')
            axs[istate].set_ylabel(state_name)
            axs[istate].legend(loc='center right')
    plt.tight_layout()
    plt.savefig(f'Fig_FIS_Demo')

    plt.show()