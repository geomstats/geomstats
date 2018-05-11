#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include "RedisClient.h"

const char JOINT_TORQUES_COMMANDED_KEY[] = "geomstats_examles::command_torques";
const char JOINT_POSITIONS_KEY[] = "geomstats_examles::joint_positions";
const char JOINT_VELOCITIES_KEY[] = "geomstats_examles::joint_velocities";


namespace gazebo
{
class iiwaControl : public ModelPlugin
{
  public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
  {
    // Store the pointer to the model
    this->model = _parent;

    model->GetJoint(joint_names[0])->SetPosition(0, 90.0/180.0*M_PI);
    model->GetJoint(joint_names[1])->SetPosition(0, -30.0/180.0*M_PI);
    model->GetJoint(joint_names[3])->SetPosition(0, 60.0/180.0*M_PI);
    model->GetJoint(joint_names[5])->SetPosition(0, -90.0/180.0*M_PI);

    command_torques.setZero(7);
    joint_pos.setZero(7);
    joint_vel.setZero(7);

    // start redis client
    HiredisServerInfo info;
    info.hostname_ = "127.0.0.1";
    info.port_ = 6379;
    info.timeout_ = { 1, 500000 };  // 1.5 seconds
    redis_client = CDatabaseRedisClient();
    redis_client.serverIs(info);

    redis_client.setEigenMatrixDerivedString(
      JOINT_TORQUES_COMMANDED_KEY, command_torques);
    redis_client.setEigenMatrixDerivedString(
      JOINT_POSITIONS_KEY, joint_pos);
    redis_client.setEigenMatrixDerivedString(
      JOINT_VELOCITIES_KEY, joint_vel);

    // Listen to the update event. This event is broadcast every
    // simulation iteration.
    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&iiwaControl::OnUpdate, this));
  }

  // Called by the world update start event
  public: void OnUpdate()
  {
    // read joint angles from simulation
    for (int i = 0; i < 7; i++)
    {
      // joint_pos(i) = model->GetJoint(joint_names[i])->Position();
      joint_pos(i) = model->GetJoint(joint_names[i])->GetAngle(0).Radian();
      joint_vel(i) = model->GetJoint(joint_names[i])->GetVelocity(0);
    }

    // write joint angles to redis
    redis_client.setEigenMatrixDerivedString(
      JOINT_POSITIONS_KEY, joint_pos);
    redis_client.setEigenMatrixDerivedString(
      JOINT_VELOCITIES_KEY, joint_vel);

    // read torques from redis
    redis_client.getEigenMatrixDerivedString(
      JOINT_TORQUES_COMMANDED_KEY, command_torques);

    // send torques to simulation
    for (int i = 0; i < 7; i++)
    {
      model->GetJoint(joint_names[i])->SetForce(0, command_torques(i));
    }
  }

  // Pointer to the model
  private: physics::ModelPtr model;

  // Pointer to the update event connection
  private: event::ConnectionPtr updateConnection;

  // list of joint names from sdf file
  std::vector<std::string> joint_names =
    {"joint_0", "joint_1", "joint_2",
    "joint_3", "joint_4", "joint_5", "joint_6"};

  // redis
  CDatabaseRedisClient redis_client;

  // command torques
  Eigen::VectorXd command_torques;

  // positions and velocities
  Eigen::VectorXd joint_pos, joint_vel;
};

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(iiwaControl)
}  // namespace gazebo
