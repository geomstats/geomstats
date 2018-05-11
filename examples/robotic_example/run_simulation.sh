export GAZEBO_PLUGIN_PATH=$(pwd)/build:$GAZEBO_PLUGIN_PATH
export GAZEBO_MODEL_PATH=$(pwd)/model:$GAZEBO_MODEL_PATH

gazebo ./simulation.world --verbose &
sleep 2.5
./build/controller 
