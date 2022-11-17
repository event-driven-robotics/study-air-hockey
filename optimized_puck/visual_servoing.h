#pragma once

#include <yarp/os/all.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/math/Math.h>
#include <yarp/sig/all.h>

#include <yarp/dev/CartesianControl.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/IControlLimits.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IControlMode.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/dev/IVelocityControl.h>

#include <iostream>
#include <mutex>
#include <cmath>
#include <tuple>
#include <numeric>
#include <vector>
#include <deque>
#include <map>

using namespace yarp::os;
using namespace yarp::sig;
using namespace std;

class PID
{
    double Kp,Ki;
    double integral;

public:
    // constructor
    PID() : Kp(0.0), Ki(0.0), integral(0.0) { }

    // helper function to set up sample time and gains
    void set(const double Kp, const double Ki)
    {
        this->Kp=Kp;
        this->Ki=Ki;
    }

    // compute the control command
    double command(const double reference, const double feedback, const double Ts)
    {
        // the actual error between reference and feedback
        double error=reference-feedback;

        // accumulate the error
        integral+=error*Ts;

        // compute the PID output
        return (Kp*error+Ki*integral);
    }

    void reset()
    {
        integral = 0;
    }
};

class headControl {

private:

    yarp::dev::PolyDriver         vel_driver, pos_driver;
    yarp::dev::IEncoders         *ienc;
    yarp::dev::IControlMode *imod;
    yarp::dev::IVelocityControl *ivel;
    yarp::dev::IPositionControl *ipos;
    yarp::dev::IControlLimits *ilim;

    int nAxes;
    std::vector<PID*> controllers;
    std::vector<double> velocity;
    std::vector<double> encs;

    int u_fixation;
    int v_fixation;

protected:

public:

    headControl() : nAxes(6), u_fixation(0), v_fixation(0) {}

    bool initVelControl(int height, int width)
    {
        yarp::os::Property option;
        option.put("device","remote_controlboard");
        option.put("remote","/icub/head");
        option.put("local","/vel_controller");

        if (!vel_driver.open(option))
        {
            yError()<<"Unable to open the device vel_driver";
            return false;
        }

        // open the views

        if(!vel_driver.view(ienc)) {
            yError() << "Driver does not implement encoder mode";
            return false;
        }
        if(!vel_driver.view(imod)) {
            yError() << "Driver does not implement control mode";
            return false;
        }
        if(!vel_driver.view(ivel)) {
            yError() << "Driver does not implement velocity mode";
            return false;
        }

        // retrieve number of axes
        int readAxes;
        ienc->getAxes(&readAxes);
        if(readAxes != nAxes) {
            yError() << "Incorrect number of axes" << readAxes << nAxes;
            return false;
        }

        velocity.resize(nAxes, 0.0);
        encs.resize(nAxes);
        controllers.resize(nAxes);
        for(int i = 0; i < nAxes; i++)
            controllers[i] = new PID;

        // set up our controllers
        controllers[0]->set(2.0, 0.0); //neck pitch
        controllers[1]->set(0.0, 0.0); //neck roll
        controllers[2]->set(2.0, 0.0); //neck yaw

        u_fixation = width / 2;
        v_fixation = height / 2;

        //set velocity control mode
        return setVelocityControl();

    }

    bool initPosControl(){
        yarp::os::Property option;
        option.put("device", "remote_controlboard");
        option.put("remote", "/icub/head");
        option.put("local", "/pos_controller");

        pos_driver.open(option);
        pos_driver.view(ipos);
        pos_driver.view(imod);
        pos_driver.view(ilim);

        return true;
    }

    bool setVelocityControl()
    {
        int naxes;
        ivel->getAxes(&naxes);
        std::vector<int> modes(naxes, VOCAB_CM_VELOCITY);

        imod->setControlModes(modes.data());

        return true;
    }

    void resetRobotHome(){

        int naxes;
        ipos->getAxes(&naxes);
        std::vector<int> modes(naxes, VOCAB_CM_POSITION);
        std::vector<double> vels(naxes, 10.);
        std::vector<double> accs(naxes, std::numeric_limits<double>::max());
        std::vector<double> poss(naxes, 0.);
        poss[2]=0;
        poss[0]=-23;

        imod->setControlModes(modes.data());
        ipos->setRefSpeeds(vels.data());
        ipos->setRefAccelerations(accs.data());
        ipos->positionMove(poss.data());

        auto done = false;
        while(!done) {
            yarp::os::Time::delay(1.);
            ipos->checkMotionDone(&done);
        }

        setVelocityControl();
    }

    void scroll_yaw(){
        int naxes;
        ipos->getAxes(&naxes);
        std::vector<int> modes(naxes, VOCAB_CM_POSITION);
        std::vector<double> vels(naxes, 20.);
        std::vector<double> accs(naxes, std::numeric_limits<double>::max());
        std::vector<double> poss(naxes, 0.);
        poss[2]=-10;
        poss[0]=5.977;

        imod->setControlModes(modes.data());
        ipos->setRefSpeeds(vels.data());
        ipos->setRefAccelerations(accs.data());
        ipos->positionMove(poss.data());

        auto done = false;
        while(!done) {
            yarp::os::Time::delay(1.);
            ipos->checkMotionDone(&done);
        }
    }

    void controlMono(int u, int v, double dt)
    {

        double neck_tilt=controllers[0]->command(v_fixation,v, dt);  // neck pitch
        double neck_pan=controllers[2]->command(u_fixation,u, dt); // neck yaw

        // send commands to the robot head
        velocity[0]=neck_tilt;          // neck pitch
        velocity[1]=0.0;                // neck roll
        velocity[2]=neck_pan;           // neck yaw
        velocity[3]=0.0;                // eyes tilt
        velocity[4]=0.0;                // eyes vers
        velocity[5]=0.0;                // eyes verg

//        yInfo()<<"vel: "<<velocity[0]<<velocity[2];
        //ivel->velocityMove(0, neck_tilt);
        //ivel->velocityMove(2, neck_pan);
        ivel->velocityMove(velocity.data());

//        double vel_tilt, vel_pan;
//        ivel->getRefVelocity(0,&vel_tilt);
//        ivel->getRefVelocity(2,&vel_pan);
//        yInfo()<<"current vel"<<vel_tilt<<vel_pan;
    }

    double computeErrorDistance(int u, int v){

        double dist=sqrt((u-u_fixation)*(u-u_fixation)+(v-v_fixation)*(v-v_fixation));

        return dist;
    }

    void controlReset()
    {
        for(int i = 0; i < nAxes; i++) {
            controllers[i]->reset();
            velocity[i] = 0;
        }
        ivel->velocityMove(velocity.data());
    }

    double getJointPos(int joint_number){
        ienc->getEncoders(encs.data());
        return encs[joint_number];
    }

    void getJointLimits(int joint_num, double* joint_min, double* joint_max){

        double min, max;
        ilim->getLimits(joint_num, &min, &max);

        *joint_min = min;
        *joint_max = max;
    }

    void closeToLimit(int joint_number){

        double joint_pos_min, joint_pos_max, current_joint;
        ienc->getEncoders(encs.data());
        if (joint_number == 0){
            joint_pos_min=-27; joint_pos_max=20;
            current_joint = encs[0];
        }
        if (joint_number == 2){
            joint_pos_min=-44; joint_pos_max=44;
            current_joint = encs[2];
        }

//        getJointLimits(joint_number, &joint_pos_min, &joint_pos_max);

        double error_min = current_joint - joint_pos_max;
        double error_max = current_joint - joint_pos_min;

        double smallTh = 3;

        if (error_min < smallTh || error_max < smallTh){
            resetRobotHome();
            yInfo()<<"joints resetted";
        }
    }



};

