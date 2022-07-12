import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd 
import numpy as np 
import os

class AlphaMoSeEnv(gym.Env):
    '''
    AlphaMoSeEnv is a custom Gym Environment
    Args:
    - PFdataPath: string, path of physical field data
    - GTimeHorizon: int, time horizon of physical field, unit: s
    - GTimeStepsize: int, time step size of physical field, unit: s
    - MeasureTime: int, time required for measurement at one location, unit: s
    - IniLoc: tuple, initial location of robot, unit: m
    - MaxVel: float, max moving velocity of robot, unit: m/s
    - CostWeight: tuple, (distance_weight, time_weight), weight to calculate reward
    - MaxStep: int, max moving step, default: 1e5
    
    Action space:
    - continuous space: 
    robot moving velocities in x and y directions
    robot moving time
    robot moving command (continue / stop)
    
    Observation space:
    - continuous space: 
    x and y coordinates of the location where the robot is
    the global time that the measurement is finished at that location
    the measured variable value at that location

    The environment is valid for both static and dynamic physical fields because their data structure are identical
    The static field data also has a temporal dimension but variable values do not variate with time

    '''

    metadata={'render.modes': ['human']}


    def __init__(self, PFdataPath, GTimeHorizon, GTimeStepSize, 
        MeasureTime, IniLoc, MaxVel, CostWeight, MaxStep=1e5):
        self.stdata=pd.read_csv(PFdataPath)
        self.global_timehorizon=GTimeHorizon
        self.global_timestepsize=GTimeStepSize
        self.measure_time=MeasureTime
        self.initial_location=IniLoc
        self.maxvelocity=MaxVel
        self.cost_weight=CostWeight
        self.max_step=MaxStep
        
        self.episode_idx=0
        self.global_time=0.0

        #get max and min of coordinates
        #in PFdata csv: 
        #first column is x coordinate, second column is y coordinate, third column is z (vertical direction) coordinate
        self.domXmin, self.domXmax = np.min(self.stdata['X']), np.max(self.stdata['X'])
        self.domYmin, self.domYmax = np.min(self.stdata['Y']), np.max(self.stdata['Y'])

        #judge whether the initial location is in the domain or not
        self._whether_in_domain(self.initial_location)

        #as the initial location is in the domain, obtain the variable value of the initial location
        self.ini_var=self._measure(self.initial_location, self.global_time)
            

        #define the action space and state space
        #move-1, Stop-0
        self.action_names=['x_velocity', 'y_velocity', 'moving time', 'Continue/Stop']
        self.action_low=np.array([-self.maxvelocity, -self.maxvelocity, 0, 0])
        self.action_high=np.array([self.maxvelocity, self.maxvelocity, 100, 1])
        self.action_space=spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float64)

        self.obs_names=['x_coordinate', 'y_coordinate', 'global_time','variable value at (x, y, t)']
        self.obs_low=np.array([self.domXmin, self.domYmin, 0, 0])
        self.obs_high=np.array([self.domXmax, self.domYmax, self.global_timehorizon, 50])
        self.observation_space=spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float64)


    def reset(self):
        self.episode_idx+=1
        self.global_time=0.0
        self.step_idx=0
        self.total_moving_distance=0
        self.total_moving_time=0
        self.action=np.zeros(len(self.action_names))
        self.obs=np.zeros(len(self.obs_names))
        for i in range(2):
            self.obs[i]=self.initial_location[i]
        self.obs[2]=self.global_time
        self.obs[3]=self.ini_var

        return self.obs


    def step(self, action):
        self.step_idx+=1
        self.action=action
        #if the moving command is 'Stop', the current episode is terminated
        if self.action[3]==0:
            done=True
            self.render()

            return done, {'Total moving distance': self.total_moving_distance,
                                            'Total moving time': self.total_moving_time,
                                            'Steps': self.step_idx-1}

        #if the moving command is 'Continue', the current episode will continue 
        else:
            self.global_time=self.global_time+self.action[2]+self.measure_time
            self.obs[2]=self.global_time
            self._take_action()
            reward=self._compute_reward()

            #the current episode will not be terminated untill the max step or the end of global time horizon is reached
            if (self.step_idx < self.max_step) and (self.global_time < self.global_timehorizon):
                done=False
            else:
                done=True
                self.render()

            return self.obs, reward, done, {'Total moving distance': self.total_moving_distance,
                                            'Total moving time': self.total_moving_time,
                                            'Steps': self.step_idx}


    def _take_action(self):
        #x=x+Ux*t
        self.obs[0]=self.obs[0]+self.action[0]*self.action[2]
        #y=y+Uy*t
        self.obs[1]=self.obs[1]+self.action[1]*self.action[2]
        #obtain variable value of point (x,y) at t_global
        self.obs[3]=self._measure((self.obs[0], self.obs[1]), self.obs[2])


    def _compute_reward(self):
        #calculate the moving distance and moving time of the current step
        moving_distance=np.sqrt((self.action[0]*self.action[2])**2+(self.action[1]*self.action[2])**2)
        moving_time=self.action[2]
        #cost is a weighted sum of moving distance and moving time
        cost=moving_distance*self.cost_weight[0]+moving_time*self.cost_weight[1]
        #total moving distance and total moving time
        self.total_moving_distance+=moving_distance
        self.total_moving_time+=moving_time
    
        return cost 

    #this function returns a variable value for a given location at a given time point
    #for spatial dimension, instead of interpolation, a variable value of the nearest point is returned as the measured value of the given location
    #for temporal dimension, interpolation result is returned as the measured value of the given location if the given time point does not match the PFdata temporal resolution
    def _measure(self, location_coor, time_point):
        #a square region with its centre being the given point and its half width being delta_mea is defined and points within the region are collected
        delta_mea=0.5
        window_data_mea=self.stdata.loc[(self.stdata['X']>(location_coor[0]-delta_mea)) & (self.stdata['X']<(location_coor[0]+delta_mea)) & 
        (self.stdata['Y']>(location_coor[1]-delta_mea)) & (self.stdata['Y']<(location_coor[1]+delta_mea))]
        #calculate distances of the collected points to the given point and insert as a new column
        window_data_mea.insert(len(window_data_mea), 'Distance', np.sqrt((window_data_mea['X']-location_coor[0])**2+(window_data_mea['Y']-location_coor[1])**2))

        Q=np.divmod(time_point, self.global_timestepsize)
        #if the given time point matches the temporal resolution, then temporal interpolation can be skipped
        if Q[1]==0:
            mea_result=window_data_mea.loc[window_data_mea['Distance']==window_data_mea['Distance'].min(), ['{}'.format(time_point)]].iloc[0,0]
            
        else:
            t_1=Q[0]*self.global_timestepsize
            t_2=(Q[0]+1)*self.global_timestepsize
            result_mid=window_data_mea.loc[window_data_mea['Distance']==window_data_mea['Distance'].min(), ['{}'.format(t_1), '{}'.format(t_2)]]
            mea_result=np.interp(time_point, (t_1, t_2), result_mid.iloc[0,:])
        
        return mea_result


    #this function judges whether the current location is in the domain or not
    #a square region with its centre being the given point and its half width being delta is defined
    #how to judge:
    #1.the given point is as a centre and the square region is divided into four quadrants
    #2.if there hava points in each quadrant, the given point is considered as 'in the domain'
    #3.if at least one quadrant is empty, the given point is considered as 'outside the domain or on the domain boundary'
    def _whether_in_domain(self, location_coor):
        delta=1.0
        judge_array=[0,0,0,0]
        window_data=self.stdata.loc[(self.stdata['X']>(location_coor[0]-delta)) & (self.stdata['X']<(location_coor[0]+delta)) & 
        (self.stdata['Y']>(location_coor[1]-delta)) & (self.stdata['Y']<(location_coor[1]+delta)), ['X', 'Y']]
        judge_array[0]=window_data.loc[(window_data['X']<location_coor[0]) & (window_data['Y']<location_coor[1])].empty
        judge_array[1]=window_data.loc[(window_data['X']<location_coor[0]) & (window_data['Y']>location_coor[1])].empty
        judge_array[2]=window_data.loc[(window_data['X']>location_coor[0]) & (window_data['Y']<location_coor[1])].empty
        judge_array[3]=window_data.loc[(window_data['X']>location_coor[0]) & (window_data['Y']>location_coor[1])].empty

        assert True not in judge_array, 'The location is outside the domain or is on the domain boundary'


    #this function exports a template according to user's request for the purpose of accuracy evaluation
    #the user can determine how many points and at which time point for accuracy evaluation
    #the function will randomly sample n points in the physical field and save as a ground truth
    #the user needs to fill in the template with their prediction results 
    def request_evauation_template(self):
        self.sampling_number=int(input('Please input sampling number:'))
        self.target_time=float(input('Please input target time:'))
        self.template_path=input('Please input a path for template export:') #e.g. C:/desktop/evaluation_template.csv

        #if the target time does not match the temporal resolution of the physical field, temporal interpolation will be executed
        Q=np.divmod(self.target_time, self.global_timestepsize)
        if Q[1]==0:
            self.sampling_info=self.stdata[['X', 'Y', '{}'.format(self.target_time)]].sample(n=self.sampling_number, random_state=1)
            sampling_template=self.sampling_info.copy(deep=True)
            sampling_template['{}'.format(self.target_time)]=''
            sampling_template.to_csv(self.template_path)

        else:
            t_1=Q[0]*self.global_timestepsize
            t_2=(Q[0]+1)*self.global_timestepsize
            self.sampling_info=self.stdata[['X', 'Y', '{}'.format(t_1), '{}'.format(t_2)]].sample(n=self.sampling_number, random_state=1)

            value_target_time=[]
            for i in range(self.sampling_number):
                interpolate_value=np.interp(self.target_time, (t_1, t_2), (self.sampling_info['{}'.format(t_1)].iloc[i], self.sampling_info['{}'.format(t_2)].iloc[i]))
                value_target_time.append(interpolate_value)
            self.sampling_info['{}'.format(self.target_time)]=value_target_time
            sampling_template=self.sampling_info.copy(deep=True)
            sampling_template.drop(['{}'.format(t_1), '{}'.format(t_2)],axis=1, inplace=True)
            sampling_template['{}'.format(self.target_time)]=''
    
            sampling_template.to_csv(self.template_path)


    #root mean square error (RMSE) between user's prediction results and the ground truth is computed as an index for accuracy evaluation
    def compute_accuracy(self):
        assert os.path.exists(self.template_path), 'Please request a template first!'

        results=pd.read_csv(self.template_path)['{}'.format(self.target_time)]
        ground_truth=self.sampling_info['{}'.format(self.target_time)]
        sum_sq=0
        for i in range(self.sampling_number):
            sum_sq+=(results.iloc[i]-ground_truth.iloc[i])**2
        rmse=np.sqrt(sum_sq/self.sampling_number)

        return rmse

    def render(self, mode='human', close=False):
        print('Episode: {}'.format(self.episode_idx))
        print('Total moving distance (m)')
        print(self.total_moving_distance)
        print('Total moving time (s)')
        print(self.total_moving_time)