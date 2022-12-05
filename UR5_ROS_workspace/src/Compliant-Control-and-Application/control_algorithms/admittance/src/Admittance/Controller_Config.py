# !/usr/bin/env python3
from filters import LowPass_Filter


class SettingsHandler(object):
    def __init__(self):
        DISABLED = 0
        ENABLED = 1
        self.filters = {}

        self.public = {}
        self.public['pos_start'] = -0.3
        self.public['pos_des'] = -0.12 #0.3  # 0.2
        # 18 tiles 10 cm each
        # -0.15 -0.09
        self.public['pos_max'] = 0.3  # 0.2
        self.public['tol_des'] = 0.001
        self.public['max_epi_duration'] = 60  # seconds

        start = self.public['pos_start']
        dpos = self.public['pos_des']-self.public['pos_start']
        ntiles = 17
        w_tile = 0.01 # meters
        dtile = ntiles*w_tile
        # self.public['f_pos2dist'] = lambda _pos: (dtile / dpos) * (_pos - start)
        self.public['f_pos2dist'] = lambda _pos: _pos
        # self.public['f_pos2dist'] =  lambda _pos: (57./0.6)*(_pos+0.3)
        #self.public['f_dist2pos'] = lambda _dist: (0.6/ 57. ) * (_dist) - 0.3


        self.admittance_node = {}
        self.admittance_node['loop_rate'] = 200
        self.admittance_node['node_handler'] = 'admittance_controller'
        self.admittance_node['D_default'] = 40.
        self.admittance_node['D_max'] = 50.
        self.admittance_node['v_max'] = 1.0
        self.admittance_node['a_max'] = 10.0
        self.admittance_node['F_lb'] = 6.0  # force deadzone lower bound
        self.admittance_node['F_ub'] = 30.0  # force deadzone lower bound
        self.admittance_node['wait4RL'] = True
        self.admittance_node['plot_on_close'] = False

        self.admittance_node['K_constrain'] = 100
        self.admittance_node['D_constrain'] = 100
        self.admittance_node['M_constrain'] = 10

        M_object = 0.0
        M_FTsensor, M_gripper = 0.3, 0.925  # mass of gripper and FT sensor
        M_ee = M_gripper + M_FTsensor  # cumulative mass of ee

        self.admittance_node['virtual_mass'] = M_ee +M_object

        self.admittance_node['y_des'] = 0.55
        self.admittance_node['z_des'] = 0.42

        self.learning_node = {}
        self.learning_node['nstep_update'] = 3
        self.learning_node['loop_rate'] = 200
        self.learning_node['node_handler'] = 'RL_status'
        self.learning_node['fixed_damping'] = [None,self.admittance_node['D_default'] ][DISABLED]
        self.learning_node['verbose'] = True  # None
        self.learning_node['crisp_actions'] = [10., 22.5, 50.]

        self.learning_node['max_epi_duration'] = self.public['max_epi_duration']
        self.learning_node['num_episodes'] = 50
        self.learning_node['num_warmup'] = 1  # high exploration steps
        self.learning_node['num_perform'] = 10  # high exploitation steps

        self.learning_node['gamma'] = 0.95  # discount factor
        self.learning_node['alpha'] = 0.2 # learining rate
        self.learning_node['min_sp'] = 0.0000001  # undirected exploration noise
        self.learning_node['max_sp'] = 0.9  # undirected exploration noise
        self.learning_node['theta'] = 10.0  # directed exploration strength
        self.learning_node['lambda'] = 0.95  # decay of eligibility traces




        FIS_settings = {}
        v_max = self.admittance_node['v_max']
        dvdt_max = self.admittance_node['a_max']
        F_max = 10.
        dFdt_max = 15.0
        FIS_settings['v_range'], FIS_settings['v_bins'] = [-v_max / 2., v_max], 4
        FIS_settings['F_range'], FIS_settings['F_bins'] = [-F_max, F_max], 4
        FIS_settings['dvdt_range'], FIS_settings['dvdt_bins'] = [-dvdt_max, dvdt_max], 4
        FIS_settings['dFdt_range'], FIS_settings['dFdt_bins'] = [-dFdt_max, dFdt_max], 4
        self.learning_node['FIS_settings'] = FIS_settings

        self.ft_node = {}
        self.ft_node['loop_rate'] = 100
        self.ft_node['host'] = '192.168.0.100'
        self.ft_node['port'] = 63351  # The same port as used by the server
        self.ft_node['publish_topic'] = 'ft300_force_torque'
        self.ft_node['node_handler'] = 'torque_force_sensor_data'
        self.ft_node['que_sz'] = 3
        self.ft_node['verbose'] = False
        self.ft_node['publish_zeros_on_error'] = True

        # ROS Topic and TF definitions
        self.topics = {}
        self.topics['base_link'] = "/base_link"
        self.topics['end_link'] = "/wrist_3_link"
        self.topics['topic_arm_command'] = '/cartesian_velocity_controller/command_cart_vel'
        self.topics['topic_arm_state'] = '/cartesian_velocity_controller/ee_state'
        self.topics['topic_wrench_state'] = self.ft_node['publish_topic']  # '/ft300_force_torque' # '/wrench'
        self.topics['topic_damping_action'] = 'RL/damping_cmd'
        self.topics['topic_settings'] = 'RL/settings'
        self.topics['topic_force_reset'] = 'RL/force_reset'
        self.topics['topic_controller_state'] = 'controller/controller_state'

        # Filters
        fs = self.admittance_node['loop_rate']
        # fc = 15
        fc = 10
        MASTER_ENABLE = True
        # self.filters['acc'] 	= LowPass_Filter(fs=fs,fc=fs/35.,enable=MASTER_ENABLE)
        # self.filters['F'] 		= LowPass_Filter(fs=fs,fc=fs/80.,enable=MASTER_ENABLE)
        # self.filters['jerk'] 	= LowPass_Filter(fs=fs,fc=fs/100.,enable=MASTER_ENABLE)
        # self.filters['dFdt'] 	= LowPass_Filter(fs=fs,fc=fs/100.,enable=MASTER_ENABLE)
        self.filters['vel'] = LowPass_Filter(fs=fs, fc=5 * fc, enable=MASTER_ENABLE)
        self.filters['acc'] = LowPass_Filter(fs=fs, fc=5*fc, enable=MASTER_ENABLE)
        self.filters['F'] = LowPass_Filter(fs=fs, fc=fc, enable=MASTER_ENABLE)
        self.filters['jerk'] = LowPass_Filter(fs=fs, fc=fc-1, enable=MASTER_ENABLE)
        self.filters['dFdt'] = LowPass_Filter(fs=fs, fc=fc-4, enable=MASTER_ENABLE)
        self.filters['damping'] = LowPass_Filter(fs=self.learning_node['loop_rate'], fc=2, enable=MASTER_ENABLE)

        """
        self.filters['acc'] 	= RT_lowpass_butter(fs=fs,fc=fs/35.,order=5,enable=True)
        #self.filters['jerk'] 	= RT_lowpass_butter(fs=fs,fc=fs/40.,order=5,enable=True)
        self.filters['F'] 		= RT_lowpass_butter(fs=fs,fc=fs/80.,order=2,enable=True) 
        self.filters['jerk'] 	= RT_lowpass_butter(fs=fs,fc=fs/100.,order=2,enable=True)
        self.filters['dFdt'] 	= RT_lowpass_butter(fs=fs,fc=fs/100.,order=2,enable=True)
        """


ALL_SETTINGS = SettingsHandler()
