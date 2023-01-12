# SLAM using particle filter 
The aim of this project was to perform particle filter based SLAM using the IMU and the LIDAR data from a THOR-OP Humanoid Robot. The IMU data avaialble was filtered and used with lidar data to perform SLAM. The lidar data is transformed into the map co-ordinates by applying suitable transformations. Based on the paricle filter approach the best particle with maximum correlation is chosen and the log odds of the map is updated. This scan-matching technique is used to update the obstacles in real-time on a gridmap as well as localize the robot in the world.


## Maps built for 4 datasets
<img src="images/final0.png?raw=true" width="300" height="300"> <img src="images/final1.png?raw=true" width="300" height="300"> 
>
<img src="images/final2.png?raw=true" width="300" height="300"> <img src="images/final3.png?raw=true" width="300" height="300">