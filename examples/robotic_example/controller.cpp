// This example application runs a controller for the IIWA

#include "Sai2Model.h"
#include "RedisClient.h"
#include "LoopTimer.h"

#include <signal.h>

#include <iostream>
#include <string>

bool runloop = true;
void stop(int) {runloop = false;}

const char robot_file[] = "./model/iiwa14/iiwa14.urdf";
const char robot_name[] = "Kuka-IIWA";

unsigned long long controller_counter = 0;

// redis keys:
// // - write:
const char JOINT_TORQUES_COMMANDED_KEY[] =
    "geomstats_examles::command_torques";
// // - read:
const char JOINT_ANGLES_KEY[] =
    "geomstats_examles::joint_positions";
const char JOINT_VELOCITIES_KEY[] =
    "geomstats_examles::joint_velocities";

// trajectory
const char DESIRED_POSITION_KEY[] =
    "geomstats_examples::desired_position";
const char DESIRED_ORIENTATION_KEY[] =
    "geomstats_examples::desired_orientation";

void sighandler(int sig)
{ runloop = false; }

int main() {
    // start redis client
    HiredisServerInfo info;
    info.hostname_ = "127.0.0.1";
    info.port_ = 6379;
    info.timeout_ = { 1, 500000 };  // 1.5 seconds
    auto redis_client = CDatabaseRedisClient();
    redis_client.serverIs(info);

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // load robots
    auto robot = new Sai2Model::Sai2Model(robot_file, false);

    // read from Redis
    redis_client.getEigenMatrixDerivedString(JOINT_ANGLES_KEY, robot->_q);
    redis_client.getEigenMatrixDerivedString(JOINT_VELOCITIES_KEY, robot->_dq);

    ////////////////////////////////////////////////
    ///    Prepare the different controllers   /////
    ////////////////////////////////////////////////
    robot->updateModel();

    int dof = robot->dof();
    Eigen::VectorXd command_torques = Eigen::VectorXd::Zero(dof);

    Eigen::MatrixXd N_prec;

    // position orientation controller
    const std::string control_link = "link_7";
    const Eigen::Vector3d control_point = Eigen::Vector3d(0, 0, -0.081);

    double kp_pos = 50.0;
    double kv_pos = 14.0;
    double kp_ori = 50.0;
    double kv_ori = 14.0;

    Eigen::Vector3d desired_position, initial_position,
        current_position, current_velocity;
    Eigen::Vector3d d_phi, current_angular_velocity;
    Eigen::Matrix3d desired_orientation, initial_orientation,
        current_orientation;

    Eigen::MatrixXd J, Lambda, Jbar, N;
    J.setZero(6, dof);
    Lambda.setZero(6, 6);
    Jbar.setZero(dof, 6);
    N.setZero(dof, dof);

    robot->position(current_position, control_link, control_point);
    initial_position = current_position;
    desired_position = current_position;
    robot->rotation(current_orientation, control_link);
    initial_orientation = current_orientation;
    desired_orientation = current_orientation;

    redis_client.setEigenMatrixDerivedString(
        DESIRED_POSITION_KEY, desired_position);
    redis_client.setEigenMatrixDerivedString(
        DESIRED_ORIENTATION_KEY, desired_orientation);

    Eigen::VectorXd task_force, task_torques;
    task_force.setZero(6);
    task_torques.setZero(dof);

    // joint controller
    Eigen::VectorXd initial_joint_position, joint_task_torques;
    initial_joint_position = robot->_q;
    joint_task_torques.setZero(dof);

    double joint_kp = 10.0;
    double joint_kv = 6.0;

    // create a loop timer
    double control_freq = 1000;
    LoopTimer timer;
    timer.setLoopFrequency(control_freq);
    timer.setCtrlCHandler(stop);    // exit while loop on ctrl-c
    timer.initializeTimer(1000000);

    // while window is open:
    while (runloop) {
        // wait for next scheduled loop
        timer.waitForNextLoop();

        // 1 - read from Redis
        redis_client.getEigenMatrixDerivedString(
            JOINT_ANGLES_KEY, robot->_q);
        redis_client.getEigenMatrixDerivedString(
            JOINT_VELOCITIES_KEY, robot->_dq);

        // 2 - update the model for robot and tasks
        robot->updateModel();
        N_prec = Eigen::MatrixXd::Identity(dof, dof);

        robot->J_0(J, control_link, control_point);
        robot->operationalSpaceMatrices(Lambda, Jbar, N, J, N_prec);
        N_prec = N;

        // 3 - Compute joint torques
        double time = controller_counter/control_freq;

        //----- Posori task
        // read desired and current position
        redis_client.getEigenMatrixDerivedString(
            DESIRED_POSITION_KEY, desired_position);
        robot->position(current_position, control_link, control_point);
        current_velocity = J.block(0, 0, 3, dof)*robot->_dq;

        // read desired and current orientation
        redis_client.getEigenMatrixDerivedString(
            DESIRED_ORIENTATION_KEY, desired_orientation);
        robot->rotation(current_orientation, control_link);
        Sai2Model::orientationError(
            d_phi, desired_orientation, current_orientation);
        current_angular_velocity = J.block(3, 0, 3, dof)*robot->_dq;

        // compute torques
        task_force.head(3) = -kp_pos*
            (current_position - desired_position) - kv_pos*current_velocity;
        task_force.tail(3) = -kp_ori*d_phi - kv_ori*current_angular_velocity;
        task_force = Lambda*task_force;
        task_torques = J.transpose() * task_force;

        //----- Joint task
        joint_task_torques = robot->_M * (
            -joint_kp*(robot->_q - initial_joint_position)
            -joint_kv*robot->_dq);

        //----- Final torques
        command_torques = task_torques + N_prec.transpose()*joint_task_torques;

        // 4 - Send control torques to redis server
        redis_client.setEigenMatrixDerivedString(
            JOINT_TORQUES_COMMANDED_KEY, command_torques);

        controller_counter++;
    }

    command_torques << 0, 0, 0, 0, 0, 0, 0;
    redis_client.setEigenMatrixDerivedString(
        JOINT_TORQUES_COMMANDED_KEY, command_torques);

    return 0;
}
