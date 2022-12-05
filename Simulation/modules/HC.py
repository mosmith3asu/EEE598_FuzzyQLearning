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

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import norm

class MinJerkOptimization():
    def __init__(self,x0,xf,duration,N):
        self.fig, self.ax = plt.subplots(3, 2)
        # Unpack Settings =========+
        self.N = N  # number of uniform timesteps to optimize over (decrease for faster computation)
        self.ns = 3                                     # number of states [pos, vel, acc]
        self.x0 = np.array(x0).reshape([self.ns, 1])    # init state
        self.xf = np.array(xf).reshape([self.ns, 1])    # final state
        self.t0 = duration[0]                           # start time of trajectory
        self.tf = duration[1]                           # end time of trajectory

        self.dt = (self.tf - self.t0) / N               # size of timestep
        self.timeseries = np.linspace(self.t0, self.tf, N)  # timeseries trajectory

        # Define human bias in trajectory generation  ============
        self.bias = {}
        self.bias['rushed'] = 0.8  # [0,1] encourages faster time to get to final position (not acc and vel)

        # Define SS control problem ============
        self.A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype='float64')
        self.B = np.array([[0, 0, 1]], dtype='float64').T

        # Optimization variables ======
        self.u0 = np.zeros(N - 1)  # guess control input @ every timestep except final
        self.const = []  # list of constraints on optimization
        self.bnds = []  # bounds/limits of the design variable J during optimization
        self.J = []  # control input minimized during optimization that satisfies end constraint

    def get_state_trajectory(self,u):
        # xdot = np.zeros([2*self.N - 1, self.ns,1])
        # xt = np.zeros([2*self.N, self.ns,1])
        xdot = np.zeros([self.N - 1, self.ns, 1])
        xt = np.zeros([self.N, self.ns, 1])
        xt[0, :] = self.x0.reshape(3,1)
        # for i, t in enumerate(range(2*self.N - 1)):
        for i, t in enumerate(range(self.N - 1)):
            if t<self.N-1:
                xdot[t] = np.dot(self.A, xt[t]) + self.B * u[t]
                xt[t + 1] = xt[t] + xdot[t] * self.dt
            else:
                xdot[t]=0;
                xt[t+1]=xt[t]
        return xt

    def objective(self,ut,wf=0.9):
        """ minimize_{ut} max{||ut||2} """
        xt = self.get_state_trajectory(ut)


        # pos_err = np.abs(self.xf[0]-xt[:,0])
        # t_pen = self.rush*np.sum(pos_err)
        # xf_pen = norm(np.abs(xt[iclosest_xf] - self.xf))      # extreme penalty for end condition at time t

        xf_pen = norm(np.abs(xt[-1] - self.xf),ord=2)           # extreme penalty for ending at end of trial
        ut = ut*self.t_discount # discount jerks for myopia

        return  (1-wf)*norm(ut,ord=2) + (wf)*np.exp(xf_pen)
        # return (1-wpen)*norm(ut)+wpen*np.exp(xf_pen) #+ penalty

    def get_jerk(self,rush=0.0):
        # Load bias params
        #self.t_discount = np.array([np.power(1-self.bias['myopia'], t) for t in range(self.N - 1)])

        if rush>0: self.t_discount = np.linspace(1+rush,1-rush,self.N-1)
        elif rush<0: self.t_discount = np.linspace(1+rush,1-rush,self.N-1)
        else: self.t_discount = np.ones(self.N-1)/(self.N-1)

        self.t_discount =  self.t_discount*(1/np.linalg.norm(self.t_discount))
        # Run opt
        result = minimize(self.objective, constraints=self.const, x0=self.u0)
        self.J = result['x']
        return self.J

    def preview(self,show=True):
        colors = ['r','b','g','m']

        xt = self.get_state_trajectory(self.J)


        ir, ic = 0, 0
        self.ax[ir, ic].plot(self.timeseries[0:-1], self.J,color=colors[-1])
        self.ax[ir, ic].set_title('Jerk Response')
        self.ax[ir, ic].set_ylabel('Jerk (J)')
        self.ax[ir, ic].set_xlabel('Time (t)')

        si = 2
        ir,ic = 0,1
        self.ax[ir, ic].plot(self.timeseries, xt[:, si],label='$a$',color=colors[si])
        self.ax[ir, ic].hlines(self.xf[si],label='Target', xmin=self.t0, xmax=self.tf, color='k', ls=':')
        self.ax[ir, ic].set_title('Acceleration Response')
        self.ax[ir, ic].set_ylabel('Acceleration (a)')
        self.ax[ir, ic].set_xlabel('Time (t)')
        self.ax[ir, ic].legend()


        si = 1
        ir, ic = 1, 0
        self.ax[ir, ic].plot(self.timeseries, xt[:, si],label='$v$',color=colors[si])
        self.ax[ir, ic].hlines(self.xf[si],label='Target', xmin=self.t0, xmax=self.tf, color='k', ls=':')
        self.ax[ir, ic].set_title('Velocity Response')
        self.ax[ir, ic].set_ylabel('Velocity (v)')
        self.ax[ir, ic].set_xlabel('Time (t)')
        self.ax[ir, ic].legend()

        si = 0
        ir, ic = 1, 1
        self.ax[ir, ic].plot(self.timeseries, xt[:, si],label='$x$',color=colors[si])
        self.ax[ir, ic].hlines(self.xf[si],label='Target', xmin=self.t0, xmax=self.tf, color='k', ls=':')
        self.ax[ir, ic].set_title('Position Response')
        self.ax[ir, ic].set_ylabel('Position (x)')
        self.ax[ir, ic].set_xlabel('Time (t)')
        self.ax[ir, ic].legend()

        # yoffs = np.linspace(0,0.3,self.N)
        ir, ic = 2, 1
        yoffs = np.zeros(self.N)
        self.ax[ir, ic].plot(xt[:, si],yoffs, label='$x$', color=colors[si],marker = 'o',markersize=3)
        # ax[ir, ic].scatter(self.xf[si], yoffs[-1], s=150, label='target', color='k', fc='w', alpha=.5,linewidths=3)
        self.ax[ir, ic].vlines(self.xf[si], ymin=-1, ymax=1, label='target', color='k', ls=':')
        self.ax[ir, ic].set_title('Linear Trajectory')
        self.ax[ir, ic].set_xlabel('Position (x)')
        self.ax[ir, ic].set_ylim([-1,1])
        self.ax[ir, ic].set_yticks([])
        self.ax[ir, ic].legend()

        plt.tight_layout()
        if show: plt.show()

class ImpedanceController():
    # def init_impedance_control(self,stiffness, damping, inertia):
    def __init__(self,stiffness, damping, inertia,xt,dt):
        self.pt = xt
        self.dt = dt
        self.Kp = stiffness
        self.Kd = damping
        self.Im = inertia #(mass)
        self.ns = 3
        self.uF = [] # force control input

        self.tauf = 1 # number of timesteps in future to track as attractor

        # Define SS
        self.A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype='float64')
        self.B = np.array([[0, 1, 0]], dtype='float64').T

        # Define stopping conditions
        self.settle_buffer = 10 # number of extra timesteps for response to settle (trimmed if settles early)
        self.tol = 0.01         # tolerance for settling accuracy
    def get_force(self,x,xhat,Fint=0):
        """ System Model:  Im * x[2] = Kp * (xhat-x) - Kd*v + uF
        :param x: [pos,vel,acc] & xhat: attractor/target [pos desired]
        :return F: force control output
        """
        uF = self.Im * x[2] + self.Kp * (xhat[0] - x[0]) - self.Kd * x[1] + Fint
        return uF

    def get_state_trajectory(self,Xhat):
        """ Run control loop for human impedance
        :param Xhat: desired timeseries trajectory
        :return:
        """
        # calc stats
        self.N = Xhat.shape[0]
        self.Tmax = self.settle_buffer * self.N         # max settling time

        x0,xf = Xhat[0],Xhat[-1]
        extra_xf = xf*np.ones([self.Tmax - Xhat.shape[0],self.ns,1]) # get extra final states
        Xhat = np.append(Xhat,extra_xf,axis=0)          # add final state in trajectory as buffer
        self.Xhat_ext = np.copy(Xhat)

        xdot = np.zeros([self.Tmax - 1, self.ns,1])     # SS response
        xt = np.empty([Xhat.shape[0],Xhat.shape[1],1])  # actual state
        uF = np.empty(Xhat.shape[0]-1)                  # force control input
        xt[0] = x0.reshape([3,1])                       # set initial state

        for t in range(self.Tmax-1):
            txhat = min(t+self.tauf, self.Tmax-1-self.tauf) # handling overflow
            uF[t] = self.get_force(xt[t],Xhat[txhat]) # impedance control input
            xdot[t] = np.dot(self.A,xt[t] ) + self.B*uF[t]
            xt[t + 1] = xt[t] + xdot[t] * self.dt
            # Break loop and trim variables if at goal
            err = xf.reshape([3,1])-xt[t-1,:] #if t>=1 else np.array([10,10,10])
            if np.all(self.tol > np.abs(err)):
                xt = xt[0:t] # xdot = xdot[0:t,:]
                self.Xhat_ext = self.Xhat_ext[0:t,:]
                uF = uF[0:t-1]
                break

            # Debugging ----
            # print(f'uF[t] = {np.round(uF[t],2)} \txdot[r]={np.round(xdot[t].flatten(),2)} '
            #       f'\t x[t] = {np.round(xt[t].flatten(),2)}\t xhat[t] = {np.round(Xhat[t],2)}')

        self.uF = uF
        return xt

def main():
    """
    ########################################################################
    Minimum Jerk Trajectory Generation
    ########################################################################
    """
    N = 50          # number of uniform timesteps to sample in optimization (decrease for computation speed)
    x0 = [0,0,0]    # start [pos,vel,acc]
    xf = [10,0,0]   # final [pos,vel,acc]
    dur = [0,5]     # [t0,tf] start and end time of trajectory
    optimizer = MinJerkOptimization(x0,xf,dur,N)

    """ MYOPIA BIAS: the amount of shortsightedness of planner
    higher myopia = higher discount u(t)*(gamma^t) where gamma = 1-myopia
    myopia = 0 indicates that no discounting of future inputs u (optimal
    myopia = 0.99 extreme discounting of future rewards
    more myopia indicates neglect for the jerk of later timesteps
    myopia \in [0,1]
    default = 0.0 
    KEEP AT RELATIVELY LOW VALUES DUE TO EXPONENTIALLY INCREASING AT EACH TIMESTEP
    """


    J = optimizer.get_jerk(rush=0.5)
    xt = optimizer.get_state_trajectory(J)
    optimizer.preview(show=True)
    #
    # J = optimizer.get_jerk(rush=-0.5)
    # xt = optimizer.get_state_trajectory(J)
    # optimizer.preview(show=False)
    #
    # J = optimizer.get_jerk(rush=0)
    # xt = optimizer.get_state_trajectory(J)
    # optimizer.preview(show=True)

    """
    ######################################################################## 
    Impedance Control Model for Human Trajectory Tracking
    ######################################################################## 
    """
    fig,axs = plt.subplots(3,1)
    stiffness = 4.0 # increase for more aggressive/accurate tracking
    damping = 0.5   # increase if unstable
    mass = 0        # !!! need to double-check calculations for this !!!!
    dt = optimizer.dt
    Hcontrol = ImpedanceController(stiffness,damping,mass,xt,dt)

    """ ATTRACTOR TIME OFFSET: how forward thinking human is when following their MJT they panned
    tau_f = 0 tracks displacement between the current state and current desired state
    tau_f > 0 tracks displacement between the current state and future desired state
    tauf \in int([0,N]) although smaller number of timestamps recommended for stability
    default = 0
    """
    Hcontrol.tauf = 0 # track current state
    Hcontrol.tauf = 0  # track  state 3 timestamps in future
    Hcontrol.tauf = 3  # track  state 3 timestamps in future
    Hcontrol.tauf = int(0.25/dt)  # track state 0.25 seconds in future

    xH = Hcontrol.get_state_trajectory(xt)

    si = 0
    xhat = Hcontrol.Xhat_ext
    axs[0].plot(Hcontrol.uF,label = '$||u_F||$', color='m') #/np.max(np.abs(Hcontrol.uF))
    axs[1].set_title('Control Input')
    axs[0].set_ylabel('Force ($uF$)')

    axs[1].plot(xhat[:,si], color='r', label='$\hat{x}_t$')
    axs[1].plot(xH[:,si], label = '$x_t$')
    axs[1].set_title('\nTrajectories')
    axs[1].set_ylabel('Pos ($x$)')

    si = 1
    xhat = Hcontrol.Xhat_ext
    axs[2].plot(xhat[:, si], color='r', label='$\hat{v}_t$')  # label=['$x$', '$v$', '$a$']
    axs[2].plot(xH[:, si], label='$v_t$')
    axs[2].set_ylabel('Vel ($v$)')

    # add legends
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()

    print(xH)
    print(f'Finished')
#
if __name__ == "__main__":
   main()