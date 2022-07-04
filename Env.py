import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd 
import numpy as np 
from scipy.interpolate import interpn
import os

class AlphaMobileSensing(gym.Env):
    '''
    AlphaMobileSensing is a custom Gym Environment
    Args:
    - STdataPath: string, path of physical field data
    - GTimeHorizon: int, time horizon of physical field, unit: s
    - GTimeStepsize: int, time step size of physical field, unit: s
    - MeasureTime: int, time required for measurement at one location, unit: s
    - IniLoc: tuple, initial location of robot, unit: m
    - MaxVel: float, max moving velocity of robot, unit: m/s
    - CostWeight: tuple, (distance_weight, time_weight), weight to calculate reward
    - MaxStep: int, max step, default: 1e5
    
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

    The environment is valid for both static and dynamic physical fields as their data structure are identical
    The variable values of the static field do not variate with time
    
    '''


    #function name 'request_template' shold be revised for clearity
    #function name 'judge location' should be revised for clearity


    def __init__(self, STdataPath, GTimeHorizon, GTimeStepSize, 
        MeasureTime, IniLoc, MaxVel, CostWeight, MaxStep=1e5):
        self.stdata=pd.read_csv(STdataPath)
        self.global_timehorizon=GTimeHorizon
        self.global_timestepsize=GTimeStepSize
        self.measure_time=MeasureTime
        self.initial_location=IniLoc
        self.maxvelocity=MaxVel
        self.cost_weight=CostWeight
        self.max_step=MaxStep
        

        self.episode_idx=0

        
        #get max and min of coordinates
        #data csv: first column is x, second column is y, third column is z (vertical direction)
        self.domXmin=np.min(self.stdata.iloc[:,0])
        self.domXmax=np.max(self.stdata.iloc[:,0])
        self.domYmin=np.min(self.stdata.iloc[:,1])
        self.domYmax=np.max(self.stdata.iloc[:,1])

        #judge whether the initial location is in the domain or not
        self._judge_location(self.initial_location)


        #as the initial location is in the domain, obtain the variable value of that location
        self.ini_var=self._measure(self.initial_location, 0.0)
            

        #define the action space and state space
        #move-1, Stop-0
        self.action_names=['x_velocity', 'y_velocity', 'moving time', 'Continue/Stop']
        self.action_low=np.array([-self.maxvelocity, -self.maxvelocity, 0, 0])
        self.action_high=np.array([self.maxvelocity, self.maxvelocity, 100, 1])
        self.action_space=spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float16)

        self.obs_names=['x_coordinate', 'y_coordinate', 'global_time','variable value at (x, y, t)']
        self.obs_low=np.array([self.domXmin, self.domYmin, 0, 0])
        self.obs_high=np.array([self.domXmax, self.domYmax, self.global_timehorizon, 50])
        self.observation_space=spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float16)


    def reset(self):
        self.episode_idx+=1
        self.global_time=0
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

            #the current episode will not be terminated untill the max step or global time horizon is reached
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

 
    #this function finds the nearest neighbor of a given value (e.g. x coordinate) in a given array (e.g. an array consists of all x coordinates in the domain)
    #the function is necessary for spatial interpolation and location judgement
    #its return includes the nearest neighbor that is larger than the given value and the nearest neighbor that is smaller than the given value
    def _find_nearest_neighbor(self, tar_val: float, tar_arr: np.array):
        self.tar_val=tar_val
        self.tar_arr=tar_arr
        #a distance array consists of distances between the given value and each element in the given array
        distance_arr=self.tar_arr-self.tar_val

        if 0.0 in distance_arr:
            return self.tar_val, self.tar_val
        else:
            pos_min=np.min(np.where(distance_arr>0, distance_arr, np.inf))+self.tar_val
            neg_max=np.max(np.where(distance_arr<0, distance_arr, -np.inf))+self.tar_val
            return pos_min, neg_max
    
    #this function executes spatial-temporal interpolation of variable value for a given location at a given time point
    #the spatial-temporal interpolation is necessary because the spatial-temporal resolution of the physical field has limitations
    #the given location and time point may exceed the limitaion of the field spatial-temporal resolution
    def _measure(self, point_coor: tuple, time_glo: float):
        self.point_coor=point_coor
        self.time_glo=time_glo
        X_min_max=self._find_nearest_neighbor(self.point_coor[0], self.stdata.iloc[:,0])
        Y_min_max=self._find_nearest_neighbor(self.point_coor[1], self.stdata.iloc[:,1])
        
        xx=np.array([X_min_max[0], X_min_max[1]])
        yy=np.array([Y_min_max[0], Y_min_max[1]])
        points=(xx, yy)

        Q=np.divmod(self.time_glo, self.global_timestepsize)
        #the given time point corresponds to the temporal resolution, thus temporal interpolation can be skipped
        if Q[1]==0:
            values=()
            for i in range(2):
                for j in range(2):
                    var_point=self.stdata.loc[(self.stdata['X']==xx[i]) & (self.stdata['Y']==yy[j]), ['{}'.format(self.time_glo)]]
                    values=np.concatenate((values, np.array(var_point.iloc[0,0])))
            #2-D spatial interpolation because four neighbor points exist        
            int_result=interpn(points, values, self.point_coor)
        
        else:
            t_1=Q[0]*self.global_timestepsize
            t_2=(Q[0]+1)*self.global_timestepsize
            timetuple=(t_1, t_2)
            int_result_mid=np.zeros(2)
            for k in range(2):
                values=()
                for i in range(2):
                    for j in range(2):
                        var_point=self.stdata.loc[(self.stdata['X']==xx[i]) & (self.stdata['Y']==yy[j]), ['{}'.format(timetuple[k])]]
                        values=np.concatenate((values, np.array(var_point.iloc[0,0])))
                #2-D spatial interpolation at nearest time points of the given time point
                int_result_mid[k]=interpn(points, values, self.point_coor)
            #temporal interpolation for the given time point 
            int_result=np.interp(self.time_glo, timetuple, int_result_mid)

        return int_result


    #this function judges whether the current location is in the domain or not
    #two ways to judge:
    #1. having inf or -inf in the return tuple indicates the point is outside the min/max coordinates of the domain
    #2. Not having inf or -inf in the return tuple but at least one neighbor point does not have a corresponding variable value (indicates this neighbor point is outside the domain)
    def _judge_location(self, location_coor: tuple):
        self.location_coor=location_coor
        x_neighbors=self._find_nearest_neighbor(self.location_coor[0], self.stdata.iloc[:,0])
        y_neighbors=self._find_nearest_neighbor(self.location_coor[1], self.stdata.iloc[:,1])
        assert (np.inf not in x_neighbors and -np.inf not in x_neighbors 
        and np.inf not in y_neighbors and -np.inf not in y_neighbors), 'The location is outside the domain'

        assert (self.stdata.loc[(self.stdata['X']==x_neighbors[0]) & (self.stdata['Y']==y_neighbors[0]), ['0']].shape==(1,1)
        and self.stdata.loc[(self.stdata['X']==x_neighbors[1]) & (self.stdata['Y']==y_neighbors[1]), ['0']].shape==(1,1)
        and self.stdata.loc[(self.stdata['X']==x_neighbors[0]) & (self.stdata['Y']==y_neighbors[1]), ['0']].shape==(1,1)
        and self.stdata.loc[(self.stdata['X']==x_neighbors[1]) & (self.stdata['Y']==y_neighbors[0]), ['0']].shape==(1,1)), 'The location is outside the domain'

    #this function exports a template according to user's request for the purpose of accuracy calculation
    #the user can determine how many points and at which time point for accuracy evaluation
    #the function will randomly sample n points in the physical field and save as ground truth
    #the user needs to fill in the template with their prediction results 
    def request_template(self):
        self.sampling_number=input('Please input sampling number:')
        self.target_time=input('Please input target time:')
        self.template_path=input('Please input a path for template export:') #e.g. 'C:/desktop/template.csv'

        #if the target time exceeds the temporal resolution of the physical field, temporal interpolation will be executed
        Q=np.divmod(self.target_time, self.global_timestepsize)
        if Q[1]==0:
            self.sampling_info=pd.DataFrame(self.stdata.loc[['X'], ['Y'], ['{}'.format(self.target_time)]]).sample(n=self.sampling_number, random_state=1)
            sampling_template=self.sampling_info.copy(deep=True)
            sampling_template['{}'.format(self.target_time)]=''
            sampling_template.to_csv(self.template_path)

        else:
            t_1=Q[0]*self.global_timestepsize
            t_2=(Q[0]+1)*self.global_timestepsize
            self.sampling_info=pd.DataFrame(self.stdata.loc[['X'], ['Y'], ['{}'.format(t_1)], ['{}'.format(t_2)]]).sample(n=self.sampling_number, random_state=1)

            value_target_time=[]
            for i in range(self.sampling_number):
                interpolate_value=np.interp(self.target_time, (t_1, t_2), (self.sampling_info['{}'.format(t_1)][i], self.sampling_info['{}'.format(t_2)][i]))
                value_target_time.append(interpolate_value)
            self.sampling_info['{}'.format(self.target_time)]=value_target_time
            sampling_template=self.sampling_info.copy(deep=True)
            sampling_template['{}'.format(self.target_time)]=''
            sampling_template.to_csv(self.template_path)


    #root mean square error (RMSE) between user's prediction results and ground truth values is as an accuracy index        
    def compute_accuracy(self):
        assert os.path.exists(self.template_path), 'Please request a template first!'

        results=pd.read_csv(self.template_path)['{}'.format(self.target_time)]
        ground_truth=self.sampling_info['{}'.format(self.target_time)]
        sum_sq=0
        for i in range(self.sampling_number):
            sum_sq+=(results[i]-ground_truth[i])**2
        rmse=np.sqrt(sum_sq/self.sampling_number)

        return rmse

        