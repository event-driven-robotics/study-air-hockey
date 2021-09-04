#include <yarp/os/Network.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Value.h>
#include <yarp/os/Property.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/Time.h>
#include <yarp/sig/Vector.h>
#include <yarp/sig/Matrix.h>
#include <yarp/math/Math.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/CartesianControl.h>
#include <yarp/dev/GazeControl.h>
#include <iostream>
#include <event-driven/all.h>
#include <opencv2/opencv.hpp>
#include <vector>
//#include <dirent.h>
//#include <sstream>
//#include <numeric>
//#include <iterator>


#define f 274.573
#define x0 139.228
//#define x0 152.228
#define y0 123.092

using namespace ev;
using namespace yarp::os;
using namespace yarp::sig;
using std::vector;
using std::string;


class flowPrediction : public RFModule, public Thread {

private:

    vReadPort< vector<AE> > input_port;
    yarp::os::BufferedPort<yarp::os::Bottle> eye_port;

    double x_eye, y_eye, z_eye, xa_eye, ya_eye, za_eye, theta_eye;
    double x_eye_prev, y_eye_prev, z_eye_prev, t_prev;
    double diff_x, diff_y, diff_z, vx, vy, vz, wx, wy, wz;
    yarp::sig::Matrix R_eye, R_eye_prev, diff_R;
    double dT;

//    BufferedPort< Vector > scope_port;

//    madgwickFilter orientation_filter;
//    velocityEstimator vel_estimate;
    double period;
    bool dof6;

    resolution res;


public:

    flowPrediction()
    {

    }

    virtual bool configure(yarp::os::ResourceFinder& rf)
    {
        //set the module name used to name ports
        setName((rf.check("name", Value("/motion-prediction")).asString()).c_str());

        res.height = rf.check("height", Value(240)).asInt();
        res.width  = rf.check("width",  Value(304)).asInt();
        period = rf.check("period", Value(0.01)).asDouble();
        dof6 = rf.check("dof6", Value(true)).asBool();

        Network yarp;
        if(!yarp.checkNetwork(2.0)) {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }

        //open io ports
        if(!input_port.open(getName() + "/AE:i")) {
            yError() << "Could not open input port";
            return false;
        }

        if(!eye_port.open(getName() + "/eye-frame:i")) {
            yError() << "Could not open input port";
            return false;
        }

//        if(!scope_port.open(getName() + "/scope:o")) {
//            yError() << "Could not open scope out port";
//            return false;
//        }

        yarp.connect("/study-air-hockey/eye",
                     "/motion-prediction/eye-frame:i",
                     "fast_tcp");

        yarp.connect("/vPreProcess/right:o",
                     "/motion-prediction/AE:i",
                     "fast_tcp");

        diff_R = yarp::math::zeros(3,3);
        R_eye = yarp::math::zeros(3,3);
        R_eye_prev = yarp::math::zeros(3,3);

        //start the asynchronous and synchronous threads
        return Thread::start();
    }

    virtual double getPeriod()
    {
        return period; //period of synchrnous thread
    }

    bool interruptModule()
    {
        //if the module is asked to stop ask the asynchrnous thread to stop
        return Thread::stop();
    }

    void onStop()
    {
        //when the asynchrnous thread is asked to stop, close ports and do
        //other clean up
        input_port.close();
        eye_port.close();
//        scope_port.close();
    }

    void getOpticalFlow(double x, double y, double d, vector<double> vel, double& flow_x, double& flow_y)
    {
        double f_inv = 1.0/f;
        flow_x = ((x - x0) * (y - y0) * f_inv)*vel[3] + -(f + (pow(x - x0, 2) * f_inv))*vel[4] + (y - y0)*vel[5];
        flow_y = (f + (pow(y - y0, 2) * f_inv))*vel[3] + (-(x - x0) * (y - y0) * f_inv)*vel[4] + -(x - x0)*vel[5];

        if (dof6){
            flow_x = flow_x + (-f/d)*vel[0] + (x-x0)/d*vel[2];
            flow_y = flow_y + (-f/d)*vel[1] + (y-y0)/d*vel[2];
        }

    }

    //synchronous thread
    virtual bool updateModule()
    {

        diff_x = x_eye - x_eye_prev;
        diff_y = y_eye - y_eye_prev;
        diff_z = z_eye - z_eye_prev;
        dT = yarp::os::Time::now() - t_prev;
        diff_R(0,0) = (R_eye(0,0) - R_eye_prev(0,0))/dT;
        diff_R(0,1) = (R_eye(0,1) - R_eye_prev(0,1))/dT;
        diff_R(0,2) = (R_eye(0,2) - R_eye_prev(0,2))/dT;
        diff_R(1,0) = (R_eye(0,0) - R_eye_prev(0,0))/dT;
        diff_R(1,1) = (R_eye(0,1) - R_eye_prev(0,1))/dT;
        diff_R(1,2) = (R_eye(0,2) - R_eye_prev(0,2))/dT;
        diff_R(2,0) = (R_eye(0,0) - R_eye_prev(0,0))/dT;
        diff_R(2,1) = (R_eye(0,1) - R_eye_prev(0,1))/dT;
        diff_R(2,2) = (R_eye(0,2) - R_eye_prev(0,2))/dT;

        vx = diff_x/dT;
        vy = diff_y/dT;
        vz = diff_z/dT;
        wx = diff_R(2,1);
        wy = diff_R(0,2);
        wz = diff_R(1,0);

        std::cout<<"vx="<<vx<<", vy="<<vy<<",vz="<<vz<<",wx="<<wx<<",wy="<<wy<<",wz="<<wz<<std::endl;

        Stamp yarpstamp_flow;
        vector<double> vel_eye_frame;
        vel_eye_frame.push_back(vx);
        vel_eye_frame.push_back(vy);
        vel_eye_frame.push_back(vz);
        vel_eye_frame.push_back(wx);
        vel_eye_frame.push_back(wy);
        vel_eye_frame.push_back(wz);

        static const int ms = 10;
        cv::Mat masker(res.height, res.width, CV_8U);
        masker.setTo(1);
        static cv::Mat debug_image(res.height, res.width, CV_32FC3);
        debug_image.setTo(cv::Scalar(0, 0, 0));
        cv::Mat arrow_image(res.height, res.width, CV_8UC3);
        arrow_image.setTo(0);
        //add any synchronous operations here, visualisation, debug out prints
        int fstamp = -1;
        int n_packets = input_port.queryunprocessed();
        for(auto i = 0; i < n_packets; i++) {

            const vector<AE> * q_flow = input_port.read(yarpstamp_flow);
            if(!q_flow || Thread::isStopping()) return false;

            for(auto &v : *q_flow) {

                if(fstamp < 0)fstamp=v.stamp;

                double flow_x, flow_y;
                getOpticalFlow(v.x, v.y, 1, vel_eye_frame, flow_x, flow_y);

                //                cv::Mat magnitude, angle, magn_norm;
                //                cv::cartToPolar(flow_x, flow_y, magnitude, angle, true);
                double magnitude = sqrt(flow_x*flow_x+flow_y*flow_y)/200.0 * 255.0;
                //yInfo()<<v.x<<v.y<<magnitude/255.0;
                if(magnitude > 255.0) magnitude = 255.0f;

                double angle = atan2(flow_y, flow_x);
                //                cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
                angle = 180.0*(angle + M_PI)/(2.0*M_PI);

                int y_hat = v.y - flow_y*vtsHelper::deltaS(v.stamp, fstamp) + 0.5;
                int x_hat = v.x - flow_x*vtsHelper::deltaS(v.stamp, fstamp) + 0.5;
                if(y_hat > 0 && y_hat < debug_image.rows-1 && x_hat > 0 && x_hat < debug_image.cols)
                    debug_image.at<cv::Vec3f>(y_hat, x_hat) = cv::Vec3f(angle, 255.0f, magnitude);

                if(masker.at<unsigned char>(v.y, v.x) == 0)
                    continue;

                int xl = std::max(v.x-ms, 0); int xh = std::min(v.x+ms, masker.cols);
                int yl = std::max(v.y-ms, 0); int yh = std::min(v.y+ms, masker.rows);

                masker(cv::Rect(cv::Point(xl, yl), cv::Point(xh, yh))).setTo(0);

                cv::line(arrow_image, cv::Point(v.x, v.y), cv::Point(v.x+flow_x*period, v.y+flow_y*period), CV_RGB(255,255,255));
                //                debug_image.at<cv::Vec3f>(v.y, v.x) = cv::Vec3f(180.0f, 255.0f, 255.0f);

            }
        }
        //perform prediction with Image Jacobian

        x_eye_prev = x_eye;
        y_eye_prev = y_eye;
        z_eye_prev = z_eye;
        R_eye_prev = R_eye;
        t_prev = yarp::os::Time::now();

        //make an image visualising output
        //build hsv image
        cv::Mat bgr, hsv8;
        debug_image.convertTo(hsv8, CV_8UC3);
        cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

        arrow_image.copyTo(bgr, arrow_image);


        cv::namedWindow("imu prediction", cv::WINDOW_NORMAL);
        cv::imshow("imu prediction", bgr);
        cv::waitKey(3);

        return Thread::isRunning();
    }

    //asynchronous thread run forever
    void run()
    {

        while (Thread::isRunning()) {

            yarp::os::Bottle *eye_status = eye_port.read();
            // position
            x_eye = eye_status->get(0).asDouble();
            y_eye = eye_status->get(1).asDouble();
            z_eye = eye_status->get(2).asDouble();
            // orientation in axis-angle representation
            xa_eye = eye_status->get(3).asDouble();
            ya_eye = eye_status->get(4).asDouble();
            za_eye = eye_status->get(5).asDouble();
            theta_eye = eye_status->get(6).asDouble();

            yarp::sig::Vector o_eye;
            o_eye.resize(4);
            o_eye[0]=xa_eye;
            o_eye[1]=ya_eye;
            o_eye[2]=za_eye;
            o_eye[3]=theta_eye;

            R_eye = yarp::math::axis2dcm(o_eye);

            std::cout<< "Input position eye frame= "<<x_eye<<" "<< y_eye << " "<<z_eye<<", orientation "<<xa_eye<<" " << ya_eye<<" "<<za_eye<<" "<<theta_eye<<std::endl;
            std::cout << "R="<<R_eye(0,0)<<" "<<R_eye(0,1)<<" "<<R_eye(0,2)<<std::endl;
            std::cout << R_eye(1,0)<<" "<<R_eye(1,1)<<" "<<R_eye(1,2)<<std::endl;
            std::cout << R_eye(2,0)<<" "<<R_eye(2,1)<<" "<<R_eye(2,2)<<std::endl;


        }

//        Stamp yarpstamp_imu;
//        //Stamp yarpstamp_flow;
//
//        int timestamp = -1, p_timestamp = -1;
//
//        while(true) {
//            auto n_packets = imu_port.queryunprocessed();
//            if(n_packets < 1) n_packets = 1;
//
//            //update the current value of the imu sensor values
//            for(auto i = 0; i < n_packets; i++) {
//                // const vector<IMUevent> * q_imu = imu_port.read(yarpstamp);
//                const vector<IMUevent> * q_imu = imu_port.read(yarpstamp_imu);
//                if(!q_imu || Thread::isStopping()) return;
//
//                for(auto &v : *q_imu)
//                {
//                    imu_state[v.sensor] = imu_helper.convertToSI(v.value, v.sensor);
//                }
//                if(p_timestamp < 0) p_timestamp = q_imu->front().stamp;
//                timestamp = q_imu->back().stamp;
//            }
//
//            //update the madgwick filter
//            double dtime = vtsHelper::deltaMS(timestamp, p_timestamp);
//            orientation_filter.setPeriod(dtime);
//            p_timestamp = timestamp;
//
//            orientation_filter.AHRSupdateIMU(imu_state[imuHelper::GYR_X],
//                    imu_state[imuHelper::GYR_Y],
//                    imu_state[imuHelper::GYR_Z],
//                    imu_state[imuHelper::ACC_X],
//                    imu_state[imuHelper::ACC_Y],
//                    imu_state[imuHelper::ACC_Z]);
//
//            //perform compensation
//            auto gp = orientation_filter.gravity_compensate(
//                        imu_state[imuHelper::ACC_X],
//                    imu_state[imuHelper::ACC_Y],
//                    imu_state[imuHelper::ACC_Z]);
//            vel_estimate.integrate(gp, dtime);
//
////            Vector &imu_out = scope_port.prepare();
////            imu_out.clear();
////            imu_out.push_back(vel_estimate.getLinearVelocity()[0]);
////            imu_out.push_back(vel_estimate.getLinearVelocity()[1]);
////            imu_out.push_back(vel_estimate.getLinearVelocity()[2]);
////            imu_out.push_back(imu_state[imuHelper::GYR_X]);
////            imu_out.push_back(imu_state[imuHelper::GYR_Y]);
////            imu_out.push_back(imu_state[imuHelper::GYR_Z]);
////            scope_port.write();




        }

};

int main(int argc, char * argv[])
{

    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setVerbose( false );
    rf.setDefaultContext( "event-driven" );
//    rf.setDefaultConfigFile( "motion-prediction.ini" );
    rf.configure( argc, argv );

    /* create the module */
    flowPrediction instance;
    return instance.runModule(rf);
}
