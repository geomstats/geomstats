# Robotic example

This is a simulation and controller of a robot to use with geomstats example robot_so3.py.
It uses [gazebo](http://gazebosim.org/) for the simulation and Stanford Robotics Lab software for the control part.

The example runs a simulation and controller of a KUKA IIWA14 robot. The controller is an operational space position and orientation controller at the end effector of the robot with a nullspace pose task that keeps the elbow up.

The controller reads the desired position and orientation from a redis server, populated by geomstat examples

## Dependencies
This application depends on :
* [Gazebo](http://gazebosim.org/) version 7
* [sai2-model](https://github.com/manips-sai-org/sai2-model/tree/geomstats_robotics_examples) (use tag geomstats_robotics_examples)
* [sai2-urdfreader](https://github.com/manips-sai-org/sai2-urdfreader/tree/geomstats_robotics_examples) (use tag geomstats_robotics_examples)
* Redis*: Redis server [brew, apt-get]
* Hiredis*: Redis minimalist client [brew, apt-get]
* Eigen3*: Linear algebra [brew, apt-get]
* TinyXML2*: XML parser [brew, apt-get]

## Build instructions 
```
mkdir build
cd build
cmake .. && make -j2
```

## run
* First, make sure the redis server is running. To run it type in a terminal
```
redis-server
```
* Second, run the simulation and the controller using the provided script
```
sh run_simulation
```
* Third, go to geomstats/examples and run the example robot_so3.py
```
cd ..
python3 robot_so3.py
```