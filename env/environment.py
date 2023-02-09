import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd 
import numpy as np 
import os

class AlphaMoSeEnv(gym.Env):
    '''
    AlphaMoSeEnv is a custom Gym Environment

    Version 2.0
    Improvements in this version:
    * Add a functionality of stationary sensing (def stationary_sensing())
    * Add an evaluation metrics Mean Absolute Error
    * Support mobile sensing with multiple robots

    Input Parameters:
    - PFdataPath: string, path of physical field data
    - PFTHorizon: int, time horizon of a physical field, unit: s
    - PFTStepsize: int, time step size of a physical field, unit: s
    - CostWeight: tuple, (distance_weight, time_weight), 
      weight between moving distance and moving time to compute reward, default: (0.5, 0.5)
    - MaxStep: int, maximum number of steps for an episode, default: 1e3
    - AgentNumber: int, number of robots utilized in mobile sensing
    - MeaDuration: int, time required by a robot to measure physical variables at a location, unit: s, (required to be specified for each robot, e.g., (5.0, 5.0, 5.0) for three robots, (5.0, ) for single robot)
    - IniLocation: tuple, initial location of a robot, unit: m, (required to be specified for each robot, e.g., ((1.0,2.0),(2.0,3.0),(3.0,3.0)) for three robots, ((1.0,2.0),) for single robot)
    - MaxSpeed: float, maximum moving speed of a robot, unit: m/s, default: 2.0, (required to be specified for each robot, e.g., (2.0, 2.0, 2.0) for three robots, (2.0, ) for single robot)
    
    
    Action space:
    - continuous space: 
    robot moving velocities in x and y directions
    robot moving time
    Execution command (continue / stop)
    
    Observation space:
    - continuous space: 
    x and y coordinates of a location where the robot is
    global time that the measurement is finished at that location
    measured variable value at that location

    The environment is valid for both static and dynamic physical fields because their data structure are identical
    The static field data also has a temporal dimension but variable values do not variate with time

    '''

    metadata={'render.modes': ['human']}


    def __init__(self, PFdataPath, PFTHorizon, PFTStepSize, 
        CostWeight=(0.5,0.5), MaxStep=1e3, AgentNumber, MeaDuration, IniLocation, MaxSpeed):
        self.stdata=pd.read_csv(PFdataPath)
        self.global_timehorizon=PFTHorizon
        self.global_timestepsize=PFTStepSize
        self.cost_weight=CostWeight
        self.max_step=MaxStep
        self.agent_number=AgentNumber
        self.measure_time=MeaDuration
        self.initial_location=IniLocation
        self.maxvelocity=MaxSpeed
       
        #judge whether each robot has its setting
        assert True not in (self.agent_number!=len(self.measure_time), self.agent_number!=len(self.initial_location), self.agent_number!=len(self.maxvelocity)), 'Missing parameter in agent setup'
        
        self.episode_idx=0
        self.agent_global_time=np.zeros(self.agent_number)

        #get max and min of coordinates
        #in PFdata csv: 
        #first column is x coordinate, second column is y coordinate, third column is z (vertical direction) coordinate
        self.domXmin, self.domXmax = np.min(self.stdata['X']), np.max(self.stdata['X'])
        self.domYmin, self.domYmax = np.min(self.stdata['Y']), np.max(self.stdata['Y'])

        #judge whether the initial location is in the domain or not
        for i in range(self.agent_number):
            self._whether_in_domain(self.initial_location[i])

        #as the initial location is in the domain, obtain the variable value of the initial location
        self.ini_var=[]
        for i in range(self.agent_number):
            self.ini_var.append(self._measure(self.initial_location[i], self.agent_global_time[i]))
        
        #define the action space and state space
        #continue-1, Stop-0
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
        self.agent_global_time=np.zeros(self.agent_number)
        self.step_idx=0
        self.agent_total_moving_distance=np.zeros(self.agent_number)
        self.agent_total_moving_time=np.zeros(self.agent_number)
        self.action=[]
        self.obs=[]
        for i in range(self.agent_number):
            self.action.append(np.zeros(len(self.action_names)))
            self.obs.append(np.zeros(len(self.obs_names)))
        
        for i in range(self.agent_number):
            for j in range(2):
                self.obs[i][j]=self.initial_location[i][j]
            self.obs[i][2]=self.agent_global_time[i]
            self.obs[i][3]=self.ini_var[i]

        return self.obs


    def step(self, action):
        self.step_idx+=1
        self.action=action
        #if the command of any agent is 'Stop', the current episode is terminated
        command=np.ones(self.agent_number)
        for i in range(self.agent_number):
            command[i]=self.action[i][3]
        
        if 0 in command:
            done=True
            self.render()

            return done, {'Total moving distance of each agent': self.agent_total_moving_distance, 
                                'Total moving time of each agent': self.agent_total_moving_time,
                                'Steps': self.step_idx-1}
        #if the moving command is 'Continue', the current episode will continue
        else:
            for i in range(self.agent_number):
                self.agent_global_time[i]=self.agent_global_time[i]+self.action[i][2]+self.measure_time[i]
                self.obs[i][2]=self.agent_global_time[i]
            self._take_action()
            reward=self._compute_reward()

            #the current episode will not be terminated untill the max step or the end of global time horizon is reached
            if (self.step_idx < self.max_step) and (np.max(self.agent_global_time) < self.global_timehorizon):
                done=False
            else:
                done=True
                self.render()

            return self.obs, reward, done, {'Total moving distance of each agent': self.agent_total_moving_distance,
                                            'Total moving time of each agent': self.agent_total_moving_time,
                                            'Steps': self.step_idx}


    def _take_action(self):
        for i in range(self.agent_number):
            #x=x+Ux*t
            self.obs[i][0]=self.obs[i][0]+self.action[i][0]*self.action[i][2]
            #y=y+Uy*t
            self.obs[i][1]=self.obs[i][1]+self.action[i][1]*self.action[i][2]
            #obtain variable value of point (x,y) at t_global of agent i
            self.obs[i][3]=self._measure((self.obs[i][0], self.obs[i][1]), self.obs[i][2])


    def _compute_reward(self):

        #moving distance / time of each agent in the current step
        agent_moving_distance=np.zeros(self.agent_number)
        agent_moving_time=np.zeros(self.agent_number)
        #sum of moving distance / time of all agents in the current step
        moving_distance=0
        moving_time=0
        #reward of each agent in the current step
        cost=[]

        for i in range(self.agent_number):
            #calculate the moving distance and moving time of the current step
            agent_moving_distance[i]=np.sqrt((self.action[i][0]*self.action[i][2])**2+(self.action[i][1]*self.action[i][2])**2)
            agent_moving_time[i]=self.action[i][2]
            self.agent_total_moving_distance[i]+=agent_moving_distance[i]
            self.agent_total_moving_time[i]+=agent_moving_time[i]
            moving_distance+=agent_moving_distance[i]  
            moving_time+=agent_moving_time[i]
            #for each agent, cost(reward) is a weighted sum of moving distance and moving time
            cost.append(agent_moving_distance[i]*self.cost_weight[0]+agent_moving_time[i]*self.cost_weight[1])
            
            #if consider all agents, then the cost (reward) will be:
        #cost=moving_distance*self.cost_weight[0]+moving_time*self.cost_weight[1]     
        
        return cost 


    #this function returns a variable value for a given location at a given time point
    #for spatial dimension, instead of interpolation, a variable value of the nearest point is returned as the measured value of the given location
    #nearest-neighbor interpolation
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
        delta=1.5
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
    def request_evaluation(self):
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


    #root mean square error (RMSE) and mean absolute error (MAE) between user's prediction results and the ground truth are computed as indexes for accuracy evaluation
    def compute_accuracy(self):
        assert os.path.exists(self.template_path), 'Please request a template first!'

        results=pd.read_csv(self.template_path)['{}'.format(self.target_time)]
        ground_truth=self.sampling_info['{}'.format(self.target_time)]
        sum_sq=0
        sum_ae=0
        for i in range(self.sampling_number):
            sum_sq+=(results.iloc[i]-ground_truth.iloc[i])**2
            sum_ae+=np.abs(results.iloc[i]-ground_truth.iloc[i])
        rmse=np.sqrt(sum_sq/self.sampling_number)
        mae=sum_ae/self.sampling_number

        return {'Root mean square error':rmse, 'Mean absolute error':mae}


    #this function allows users to set a sensor network to retrieve stationary sensing results which can be used to compare with mobile sensing results 
    #users are required to specify sensor number, coordinates of each sensor, and a sampling interval
    def stationary_monitoring(self):
        sensor_number=int(input('Please input stationary sensor number number:'))
        sensor_coor=np.zeros((sensor_number,2))
        sensor_log=['time']
        for i in range(sensor_number):
            sensor_coor[i,0]=float(input('Please input X coordinate of sensor #{}:'.format(i+1)))
            sensor_coor[i,1]=float(input('Please input Y coordinate of sensor #{}:'.format(i+1)))
            sensor_log.append('sensor #{}'.format(i+1))
        df_sensor_coor=pd.DataFrame(sensor_coor, index=np.arange(1,sensor_number+1,1),columns=['X coordinate', 'Y coordinate'])

        sampling_interval=float(input('Please input sampling interval:'))
        sampling_time=np.arange(0, self.global_timehorizon, sampling_interval)

        sampling_log=np.zeros((len(sampling_time), sensor_number+1))
        for i in range(len(sampling_time)):
            sampling_log[i,0]=sampling_time[i]
            for j in range(sensor_number):
                sampling_log[i,j+1]=self._measure((sensor_coor[j,0], sensor_coor[j,1]), sampling_time[i])

        stationary_monitoring_results=pd.DataFrame(sampling_log, columns=sensor_log)

        return df_sensor_coor, stationary_monitoring_results


    def render(self, mode='human', close=False):
        print('Episode: {}'.format(self.episode_idx))
        print('Total moving distance of each agent (m)')
        print(self.agent_total_moving_distance)
        print('Total moving time of each agent (s)')
        print(self.agent_total_moving_time)