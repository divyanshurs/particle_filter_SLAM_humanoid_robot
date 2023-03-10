
import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
import sys

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)
        s.free_map = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        #num_cells = s.cells.shape[0]-1
        #ret_arr = np.zeros((len(x), 2))

        flag = True

        if(flag == True):
        
            curr_x = np.clip(x, s.xmin, s.xmax)
            curr_y = np.clip(y, s.ymin, s.ymax)

            # grid_x = (np.abs(curr_x-s.xmin)/(s.xmax - s.xmin))*num_cells
            # grid_y = (np.abs(curr_y-s.ymin)/(s.ymax - s.ymin))*num_cells
            grid_x = (curr_x - s.xmin) / s.resolution
            grid_y = (curr_y - s.ymin) / s.resolution
            #print(grid_x.shape)
            #print(np.vstack((grid_x, grid_y)).shape)
            blabla = np.vstack((grid_x, grid_y)).T
            #print(blabla.shape)
            #sys.exit()
            ret_arr = np.array(blabla, dtype = int)

            return ret_arr

        # for q in range(len(x)):
        #     curr_x = np.clip(x[q], s.xmin, s.xmax)
        #     curr_y = np.clip(y[q], s.ymin, s.ymax)
        #     grid_x = int((np.abs(curr_x-s.xmin)/(s.xmax - s.xmin))*num_cells)
        #     grid_y = int((np.abs(curr_y-s.ymin)/(s.ymax - s.ymin))*num_cells)
        #     ret_arr[q] = np.array([grid_x, grid_y], dtype = int).reshape((2,))
        # print(ret_arr[88,:])
        # sys.exit()
        # return ret_arr


class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        #s.Q = 1e-8*np.eye(3)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))


    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX
        n = np.array(p).shape[1]
        r = np.random.uniform(0, 1/n)

        c = w[0]
        i = 0
        #print(w.shape)
        w_new = np.ones(w.shape[0], dtype=float)/n
        pose_new = deepcopy(p)
        for m in range(n):
            u = r+(m/n)
        
            while(u>c):
                i+=1
                c += w[i]

            #w_new[m] = w_new[i]
            pose_new[:, m] = p[:, i]
        return pose_new, w_new


        

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        #considering head as pitch and neck as yaw

        correct_ind = np.where((d <= s.lidar_dmax) & (d >= s.lidar_dmin));
        angles = angles[correct_ind];
        d = d[correct_ind];



        worldlist = np.zeros((len(d),3))
        body_lidar_disp = np.array([0,0,s.lidar_height])
        pose_yaw = p[2]
        pose_x = p[0]
        pose_y = p[1]
        world_body_disp = np.array([pose_x,pose_y,s.head_height])
        rot_mat_body = euler_to_se3(0 , head_angle, neck_angle, body_lidar_disp)
        rot_mat_world = euler_to_se3(0 , 0, pose_yaw, world_body_disp)

        new_flag = True

        if(new_flag== True):


            # 1. from lidar distances to points in the LiDAR frame
            eff_list = np.zeros((3, len(d)))
            eff_list[0,:] = d*np.cos(angles)
            eff_list[1,:] = d*np.sin(angles)
            eff_list_homog = make_homogeneous_coords_3d(eff_list)
            
            # 2. from LiDAR frame to the body frame

            new_arr = np.matmul(rot_mat_body, eff_list_homog)

            # 3. from body frame to world frame

            final_arr_homog = np.matmul(rot_mat_world, new_arr)
            final_arr = final_arr_homog[0:3,:]
            worldlist = final_arr.T


            return worldlist


        
        # 2. from LiDAR frame to the body frame

        # #TOO SLOW CONVERT TO NP
        # for bla in range(len(d)):

        #     # if(d[bla]<= s.lidar_dmin or d[bla]>= s.lidar_dmax):
        #     #     return  #CHECK

        # # 1. from lidar distances to points in the LiDAR frame
        #     theta = angles[bla]
        #     pointx = d[bla]*np.cos(theta)
        #     pointy = d[bla]*np.sin(theta)
        #     pointz = 0


        # # 2. from LiDAR frame to the body frame
        #     #would need to make a rotation matrix mostly to account for both the neck and head
        #     #considering head as pitch and neck as yaw
        #     point = np.array([pointx, pointy, pointz]).reshape((3,1))
        #     arr = make_homogeneous_coords_3d(point)
        #     new_arr = np.matmul(rot_mat_body, arr)            


        # # 3. from body frame to world frame
        #     #only translation matrix for head to ground height
        #     final_arr = np.matmul(rot_mat_world, new_arr)
        #     worldlist[bla] = final_arr[:3,:].reshape((3,))

        # print(worldlist[88,:])
        # sys.exit()
        # return worldlist

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX
        curr_pose = s.lidar[t]['xyth']
        prev_pose = s.lidar[t-1]['xyth']
        ret = smart_minus_2d(curr_pose, prev_pose)
        return ret
        
    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        control = s.get_control(t)
         #CHECK THIS
        # perturbation=[1,1,1]
        
        
        for i in range(np.array(s.p).shape[1]):
            perturbation= np.random.multivariate_normal([0,0,0], s.Q)
            # print(s.p[:,i])
            s.p[:,i] = smart_plus_2d(s.p[:,i], control) #adding control
            s.p[:,i] = smart_plus_2d(s.p[:,i], perturbation)
        # if(t==10):
            # print("sd")
        # sys.exit()

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        p = np.log(w) + obs_logp
        q = slam_t.log_sum_exp(p)
        r = np.exp(p-q)
        return r

    @staticmethod
    def multidim_intersect(arr1, arr2):
        arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
        arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
        intersected = numpy.intersect1d(arr1_view, arr2_view)
        return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        
        # print(len(s.lidar_angles))
        # print(len(s.lidar[t]['scan']))

        obs_logprob = np.zeros((np.array(s.p).shape[1],))
        t_real = s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])
        dis_arr = s.lidar[t]['scan']
        head_ang_t = s.joint['head_angles'][1][t_real]
        neck_ang_t = s.joint['head_angles'][0][t_real]
        grid_arr = []

        for k in range(np.array(s.p).shape[1]):
            wlist = s.rays2world(s.p[:,k], dis_arr,  head_ang_t, neck_ang_t, s.lidar_angles)
            x = wlist[:,0]
            y = wlist[:,1]
            array= s.map.grid_cell_from_xy(x,y)
            array = np.array(array, dtype = int)
            grid_arr.append(array)
            obs_logprob[k] = np.sum(s.map.cells[array[:,0], array[:,1]])
            # print(k)
            # print(obs_logprob)

        updated_w = s.update_weights(s.w, obs_logprob)
        s.w = updated_w
        #print(s.w)


        index = np.argmax(updated_w) #got the particle with max weight will now update the map and log map
        # print("print")
        # print(index)
        s.max_p = s.p[:, index]


        max_p_grid = s.map.grid_cell_from_xy(s.max_p[0], s.max_p[1])
        
        grid_cells_of_int = grid_arr[index]
        #updating the log odds map and will be using that to update the 
        #map
        grid_x = grid_cells_of_int[:,0]
        grid_y = grid_cells_of_int[:,1]
        #Update only where the robot is and the obstacle is wrt the robot
        s.x_free = np.ndarray.flatten(np.linspace([max_p_grid[0][0]]*len(grid_x), grid_x-1, endpoint=False).astype(int))
        s.y_free = np.ndarray.flatten(np.linspace([max_p_grid[0][1]]*len(grid_y), grid_y-1, endpoint=False).astype(int))
        #keep track of free space to make plot easy
        s.map.free_map[s.x_free, s.y_free] = 1
        s.map.log_odds[s.x_free, s.y_free] += s.lidar_log_odds_free*0.35
        s.map.log_odds[grid_x, grid_y] += s.lidar_log_odds_occ 
        np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)


        s.map.cells[s.map.log_odds >= s.map.log_odds_thresh] = 1
        s.map.cells[s.map.log_odds < s.map.log_odds_thresh] = 0

        s.resample_particles()
     
            
        

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            #print("resample_particles")
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
