#!/usr/bin/env python3
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from modules.FQL import FQLhandler
from modules.Environment import Environment
from modules.VirtualController import VirtualController

import os
import rospy


from Controller_Config import ALL_SETTINGS
SETTINGS = ALL_SETTINGS.learning_node
FILTERS = ALL_SETTINGS.filters
TOPICS = ALL_SETTINGS.topics
PUBLIC = ALL_SETTINGS.public

def main():

    try: os.mkdir(PUBLIC['DIRECTORY'])
    except: print(f"Directory {PUBLIC['DIRECTORY']} exists")
    fname = PUBLIC['DIRECTORY'] + PUBLIC['FILE_NAME']

    ########################################################
    # IMPORT CONFIG ########################################
    # Get public settings
    tol = PUBLIC['tol_des']
    goal_position = PUBLIC['pos_des']
    start_pos = PUBLIC['pos_start']
    f_pos2dist = PUBLIC['f_pos2dist']

    # get learning settings
    node_handler = SETTINGS['node_handler']
    Cd_fixed = SETTINGS['fixed_damping']
    verbose = SETTINGS['verbose']
    exp_fs = SETTINGS['loop_rate']						# expected controller sample freq
    exp_dt = 1/exp_fs										# expected controller timestep

    max_epi_duration = SETTINGS['max_epi_duration'] # seconds
    num_episodes = SETTINGS['num_episodes']
    n_warmup = SETTINGS['num_warmup']
    n_perform = SETTINGS['num_perform']
    min_sp = SETTINGS['min_sp']
    max_sp = SETTINGS['max_sp']
    
    alpha = SETTINGS['alpha']
    gamma = SETTINGS['gamma']
    lam = SETTINGS['lambda']
    sp = SETTINGS['min_sp']
    theta = SETTINGS['theta']
    n_step_update =  SETTINGS['nstep_update']

    ########################################################
    # initialize ROS objects ###############################
    rospy.init_node(node_handler, anonymous=True)  # start ROS node
    print(f'Initializing virtual controller...')
    controller = VirtualController()
    controller.load_settings(goal_position,max_epi_duration,tol)

    loop_rate = rospy.Rate(exp_fs)					# init ROS node freq
    t_last_sample = time.time()

    ########################################################
    # initialize learning objects ##########################
    env_settings = {
        'desired_pos': goal_position,
        'timestep': exp_fs,
        'start_pos': start_pos,
        'pos2dist': f_pos2dist
        }
    env = Environment(**env_settings)
    model = FQLhandler(alpha,gamma,lam,sp,theta)


    # initialize episode trackers
    stats = {}
    stats['epi_reward'] = []
    stats['epi_length'] = []
    stats['epi_energy'] = []
    stats['epi_jerk'] = []
    

    # Define exploration scheduluing
    episode_sp = np.linspace(max_sp, min_sp, num_episodes - (n_warmup + n_perform))
    episode_sp = np.hstack([max_sp * np.ones(n_warmup), episode_sp, min_sp * np.ones(n_perform)])
    
    ###########################################################
    ################# START LEARNING ALG ######################
    ###########################################################
    
    for ith_episode in range(num_episodes):
        if rospy.is_shutdown():
            plt.ioff()
            plt.close()
            break
        # Add new data point
        stats['epi_reward'].append(0)
        stats['epi_length'].append(0)
        stats['epi_energy'].append(0)
        stats['epi_jerk'].append(0)
        
        # update episode settings and initialize vars
        model.w_undir_explore = episode_sp[ith_episode]
        
        rtau_buffer = []            # reward buffer for intermediate rewards
        Jtau_buffer = []            # jerk buffer for intermediate rewards
        dtau_buffer = []            # jerk buffer for intermediate rewards
        cumR = 0                    # cumulative episode reward
        cumJerk = 0                 # cumulative jerk
        Energy = 0                  # energy in system
        
        Xt0 = env.reset()            # reset environment
        controller.reset()
        Ut0 = model.get_action(Xt0)  # select action from policy
        Cd = model.eval_Ut(Ut0)     # get global continous action
        Cd = Cd if Cd_fixed is None else Cd_fixed

        #####################################################
        # START ITERATION ###################################
        status = 'waiting'
        _rtau1 = 0

        print(f'\n\n\n\n')
        print(f'#################################################################')
        print(f'################### START OF TRIAL ##############################')
        print(f'#################################################################')
        while not controller.done:
            if rospy.is_shutdown():
                plt.ioff()
                plt.close()
                break
            """
            pos = controller.state['pos']
            vel = controller.state['vel']
            acc = controller.state['acc']
            F = controller.state['F']
            jerk = controller.state['jerk']
            dFdt = controller.state['dFdt']
            dt = controller.state['dt']
            tau = controller.state['timestamp']
            """
            while status == 'waiting':
                if rospy.is_shutdown(): break
                env.reset()
                _, _, status = env.step(controller.state,Cd) # observe env step and get reward
                env.start_time = controller.state['timestamp']
                print(f"\r\t | [{int(controller.state['timestamp'])}] waiting",end = '')
                loop_rate.sleep()
                if controller.done: break

            Xt1, _rtau1, status = env.step(controller.state, Cd)  # observe env step and get reward
            tau = env.full_state['timestamp']
            done = controller.done

            rtau_buffer.append(_rtau1)
            Jtau_buffer.append(env.full_state['jerk'])
            dtau_buffer.append(env.full_state['dt'])
            #####################################################
            # SAMPLE RL ALGORITHM @ LOWER FREQ ##################
            # if time4update(itau):
            if len(rtau_buffer) >= n_step_update:
                # update RL algorithm
                rt1 = model.memory2reward(rtau_buffer)
                Jt1 = np.sum(np.abs(Jtau_buffer)*np.array(dtau_buffer))
                et0 = model.update_EligibiltyTrace(Ut0)
                model.update_Q(et0, Xt0, Ut0, Xt1, rt1)
                
                # update episode stat variables
                Cd = model.eval_Ut(Ut0)
                Cd = Cd if Cd_fixed is None else Cd_fixed
                cumR += rt1
                cumJerk += Jt1
                Energy = env.full_state['energy']
                
                # close old learning step
                if verbose:
                    d = 3
                    print(f'\r[e={ith_episode} t={round(tau,d)} dt = {round(env.full_state["dt"],4)}] ',end='')
                    print(f'[sp = {round(model.w_undir_explore, d)} ', end='')
                    print(f't0= {round(env.start_time, d)}] ', end='')
                    print(f'theta = {round(model.w_dir_explore, d)}] ', end='')
                    print(f'rtau = {round(_rtau1,d)} ',end='')
                    print(f'Rt = {round(cumR, d)} ', end='')
                    print(f'Cd = {round(Cd,d)} ', end='')
                    # print(f'Cd = {Cd} ', end='')
                    print(f'Q = {[round(np.min(model.q), d),round(np.max(model.q), d)]} ', end='')
                    #print(f'[V = {round(env.full_state["vel"],2)} ', end='')
                    print(f'')

                # Open new learning step
                rtau_buffer = []  # reward buffer for intermediate rewards
                Jtau_buffer = [] # jerk buffer for intermediate rewards
                dtau_buffer = []
                Xt0 = Xt1  # update current state
                Ut0 = model.get_action(Xt0)  # select action from policy
                env.memory_update()
                # Close this t-timestep =============================


            stats['epi_reward'][-1] = cumR
            stats['epi_length'][-1] = tau
            stats['epi_energy'][-1] = Energy
            stats['epi_jerk'][-1] = cumJerk

            loop_rate.sleep()
            if done: break         
            # Close this tau-timestep =============================

        # plt.close()
        print(f'\t\t || R_epi = {stats["epi_reward"][-1]}')
        print(f'\t\t || E_epi = {stats["epi_energy"][-1]}')
        print(f'\t\t || J_epi = {stats["epi_jerk"][-1]}')
        env.render(epi_reward=stats['epi_reward'],
                   epi_energy=stats['epi_energy'],
                   epi_length=stats['epi_length'],
                   epi_jerk=stats['epi_jerk'])
        path = fname + f'data_episode{ith_episode}' + '.npz'
        env.memory_save(path=path,
                        epi_reward=stats["epi_reward"],
                        epi_length=stats["epi_length"],
                        epi_energy=stats["epi_energy"],
                        epi_jerk=stats["epi_jerk"]
                        )
        # CLOSE EPISODE ===============================================

    plt.ioff()
    env.render()
    env.memory_save(path = path,
                    epi_reward = stats["epi_reward"],
                    epi_length = stats["epi_length"],
                    epi_energy = stats["epi_energy"],
                    epi_jerk = stats["epi_jerk"]
                    )
    plt.show()
    # Close FQL ========================================================
#
# def main_old():
#     # RL Settings
#     ENABLE_PLOT = True
#     FIX_DAMPING = False
#
#     exp_fs = SETTINGS['loop_rate']						# expected controller sample freq
#     exp_dt = 1/exp_fs										# expected controller timestep
#     rospy.init_node(SETTINGS['node_handler'], anonymous=True) 	# start ROS node
#     loop_rate = rospy.Rate(exp_fs)					# init ROS node freq
#     t_last_sample = time.time()
#
#
#     # Controller Settings
#     goal_position = PUBLIC['pos_des']
#     start_position = PUBLIC['pos_start']
#     last_position = start_position
#
#     max_epi_duration = SETTINGS['max_epi_duration'] # seconds
#     num_episodes = SETTINGS['num_episodes']
#     tol = PUBLIC['tol_des']
#     n_step_update =  SETTINGS['n_step_update']
#
#
#     print(f'Initializing virtual controller...')
#     controller = VirtualController()
#     controller.load_settings(goal_position,max_epi_duration,tol)
#     D_DEFAULT = 10.
#
#
#     memory = MemoryHandler()
#
#     Energy = 0
#     v0 = 0.0
#     v_dot0 = 0.0
#     F0 = 0.0
#     F_dot0 = 0.0
#     Yousef = RL(v0,v_dot0,F0,F_dot0)
#
#     reward_list=[]
#
#     for ith_episode in range(num_episodes):
#         if rospy.is_shutdown(): break
#         print(f'\n\n\nStarting Epi={ith_episode}')
#         controller.reset()
#
#         v,dvdt,F,dFdt,Jerk0 = controller.get_state()
#         Yousef.__init__(v,dvdt,F,dFdt)
#         d,r=Yousef.apply_first_action(v,dvdt,F,dFdt,Jerk0)
#         #d = FILTERS['damping'].sample(d)
#
#         comu_reward=r;
#         t_last_sample = time.time()
#         # ['Episode Reward','Episode Length','Damping','Velocity']print(f'theta = {round(model.w_dir_explore, 2)}] ', end='')

#
#         memory.add('Episode Reward',comu_reward)
#         memory.add('Episode Length', 0)
#         memory.add('Episode Energy', 0)
#         memory.add('Damping', d)
#         memory.add('Velocity',Jerk0)
#         memory.add('Force',F)
#         #memory.add('Jerk',Jerk0)
#
#         memory.reset()
#         memory.save()
#         Jmemory = []
#
#
#         #if FIX_DAMPING: d=D_DEFAULT ## DUBUG ##
#         #controller.set_damping(d)
#
#         if FIX_DAMPING: controller.set_damping(D_DEFAULT) ## DUBUG ##
#         else: controller.set_damping(d)
#
#
#         #watchdog = {'start': time.time(), 'trigger': 1.1*max_epi_duration}
#         #while controller.state['timestamp'] < max_epi_duration:
#         while not controller.done:
#             if rospy.is_shutdown(): break
#             print(f"\r iter: t={controller.state['timestamp']} Rc = {comu_reward}",end='')
#             # GET CURRENT STATE
#             v,dvdt,F,dFdt ,Jerk = controller.get_state()
#             Jmemory.append(Jerk) #Jerk0=Jerk0+Jerk
#
#             # CHECK TERMINAL CONDITION
#             if controller.done: break
#
#             ####################################################
#             # SAMPLE RL ALGORITHM @ LOWER FREQ ##################
#             if len(Jmemory) >= n_step_update: # EVERY 10 ITER
#                 #Jerk0=Jerk0/10 #\n d,r=Yousef.apply_action(v,dvdt,F,dFdt,r,Jerk0)
#                 Jc = np.mean(Jmemory)
#                 #Jc = np.linalg.norm(Jmemory,ord=np.inf) #
#                 d,r=Yousef.apply_action(v,dvdt,F,dFdt,r,Jc)
#                 #d = FILTERS['damping'].sample(d)
#                 comu_reward=comu_reward+r
#                 Jmemory = [] # Jerk0=0
#
#
#
#                 ######################################################
#             # ACT ACCORDING TO POLICY
#
#             reward_list.append(comu_reward)
#             if FIX_DAMPING: controller.set_damping(D_DEFAULT) ## DUBUG ##
#             else: controller.set_damping(d)
#
#             d,_r = Yousef.select_action(v,dvdt,F,dFdt,Jerk)
#             #d = FILTERS['damping'].sample(d)
#
#             ###### ADD ENERGY CALC ###########
#             dx = pos2dist(controller.state['pos'])-last_position
#             Energy = Energy + abs(F)*dx
#
#             # Record timestep
#             memory.replace('Episode Reward', comu_reward)
#             memory.replace('Episode Length', controller.state['timestamp'] )
#             memory.replace('Episode Energy', Energy)
#             memory.add('Damping', d)
#             memory.add('Velocity',v)
#             memory.add('Force',F)
#             #memory.add('Jerk',Jerk)
#
#
#
#             loop_rate.sleep()
#         ################# END EPISODE ########
#         #"""
#         now = memory.get('Episode Length')
#
#         iax = memory.geti('Damping')
#         ydata = memory.get('Damping')
#         xdata = np.arange(len(ydata))
#         line = lines[iax]
#         line.set_xdata(xdata)
#         line.set_ydata(ydata)
#         print(f'Damping: [{len(xdata)},{len(ydata)}]')
#         line.axes.set_ylim([min(ydata), max(ydata)])
#         line.axes.set_xlim([min(xdata), max(xdata)])
#
#         iax = memory.geti('Velocity')
#         ydata = memory.get('Velocity')
#         xdata = np.arange(len(ydata))
#         line = lines[iax]
#         line.set_xdata(xdata)
#         line.set_ydata(ydata)
#         print(f'Velocity: [{len(xdata)},{len(ydata)}]')
#         line.axes.set_ylim([min(ydata), max(ydata)])
#         line.axes.set_xlim([min(xdata), max(xdata)])
#
#         iax = memory.geti('Force')
#         ydata = memory.get('Force')
#         xdata = np.arange(len(ydata))
#         line = lines[iax]
#         line.set_xdata(xdata)
#         line.set_ydata(ydata)
#         print(f'Velocity: [{len(xdata)},{len(ydata)}]')
#         line.axes.set_ylim([min(ydata), max(ydata)])
#         line.axes.set_xlim([min(xdata), max(xdata)])
#         """
#         iax = memory.geti('Jerk')
#         ydata = memory.get('Jerk')
#         xdata = np.arange(len(ydata))
#         line = lines[iax]
#         line.set_xdata(xdata)
#         line.set_ydata(ydata)
#         line.axes.set_ylim([min(ydata), max(ydata)])
#         line.axes.set_xlim([min(xdata), max(xdata)])
#         """
#
#         iax = memory.geti('Episode Reward')
#         ydata = memory.get('Episode Reward')
#         xdata = np.arange(len(ydata))
#         line = lines[iax]
#         line.set_xdata(xdata)
#         line.set_ydata(ydata)
#         line.axes.set_ylim([min(ydata), max(ydata)])
#         line.axes.set_xlim([min(xdata), max(xdata)])
#
#         iax = memory.geti('Episode Length')
#         ydata = memory.get('Episode Length')
#         xdata = np.arange(len(ydata))
#         line = lines[iax]
#         line.set_xdata(xdata)
#         line.set_ydata(ydata)
#         line.axes.set_ylim([min(ydata), max(ydata)])
#         line.axes.set_xlim([min(xdata), max(xdata)])
#
#
#         iax = memory.geti('Episode Energy')
#         ydata = memory.get('Episode Energy')
#         xdata = np.arange(len(ydata))
#         line = lines[iax]
#         line.set_xdata(xdata)
#         line.set_ydata(ydata)
#         line.axes.set_ylim([min(ydata), max(ydata)])
#         line.axes.set_xlim([min(xdata), max(xdata)])
#
#
#         fig.canvas.flush_events()
#         fig.canvas.draw()





if __name__=="__main__":
    main()
    

