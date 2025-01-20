/*
 *   Copyright (C) 2021 Event-driven Perception for Robotics
 *   Author: luna.gava@iit.it
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "puck_position.h"

using namespace ev;
using namespace cv;
using namespace yarp::os;
using namespace yarp::sig;
using namespace std;

bool puckPosModule::configure(yarp::os::ResourceFinder& rf) {

    // options and parameters
    w = rf.check("w", Value(640)).asInt32();
    h = rf.check("h", Value(480)).asInt32();
    n_trial = rf.check("n_trial", Value(1)).asInt8();
    n_exp = rf.check("n_exp", Value(1)).asInt8();

    // module name
    setName((rf.check("name", Value("/puck_position")).asString()).c_str());

    if (!input_port.open(getName() + "/AE:i"))
        return false;

    if(!image_out.open(getName() + "/image:o")){
        yError()<<"Can't open output image port for visualization";
        return false;
    }

    EROS_vis.init(w, h, 7, 0.1);
//    EROS_vis.init(w, h, 5, 0.3);

    yarp::os::Network::connect("/atis3/AE:o", getName("/AE:i"), "fast_tcp");

    cv::Mat temp = EROS_vis.getSurface();
    eros_thread.initialise(temp, 19, cv::Rect(60, 150, 500, 80), 1000, &m2, n_trial, n_exp);
    eros_thread.start();

//    pause = false;

//    cv::namedWindow("RESULT", cv::WINDOW_AUTOSIZE);
//    cv::moveWindow("RESULT", 300,300);
//
//    cv::namedWindow("DETECT_MAP", cv::WINDOW_NORMAL);
//    cv::moveWindow("DETECT_MAP", 500,500);
//
//    cv::namedWindow("DETECT_HEAT_MAP", cv::WINDOW_NORMAL);
//    cv::moveWindow("DETECT_HEAT_MAP", 500,500);
//
    cv::namedWindow("ROI TRACK", cv::WINDOW_NORMAL);
    cv::moveWindow("ROI TRACK", 600,600);

    cv::namedWindow("ZOOM", cv::WINDOW_NORMAL);
    cv::moveWindow("ZOOM", 800,1200);

    cv::namedWindow("GAUSSIAN MUL", cv::WINDOW_NORMAL);
    cv::moveWindow("GAUSSIAN MUL", 600,1200);

    cv::namedWindow("FINAL TRACK", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("FINAL TRACK", 0,0);

//    cv::namedWindow("GAUSSIAN MUL", cv::WINDOW_NORMAL);
//    cv::moveWindow("GAUSSIAN MUL", 1400,1400);
//
//    cv::namedWindow("ell_filter", cv::WINDOW_NORMAL);
//    cv::moveWindow("ell_filter", 320,320);

//    cv::namedWindow("init filter", cv::WINDOW_NORMAL);
//    cv::moveWindow("init filter", 320,320);

    yarp::os::Network::connect(getName("/image:o"), "/puckView", "fast_tcp");

    return Thread::start();
}

void puckPosModule::run() {

    double time_offset = -1.0;
    ev::info read_stats = input_port.readChunkN(1);
    if(input_port.isStopping()) return;
    // time_offset = input_port.begin().packetTime();
    eros_thread.setFirstTime(time_offset);

    while (Thread::isRunning()) {

        input_port.readChunkN(1);
        if(input_port.isStopping())
            break;
        for(auto a = input_port.begin(); a != input_port.end(); a++) {

            if ((*a).x>0 && (*a).x < 640 && (*a).y>0 && (*a).y < 480){
                EROS_vis.update((*a).x, (*a).y);

                yarp::sig::PixelBgr &ePix = puckMap.pixel((*a).x,(*a).y);
                ePix.r = ePix.b = ePix.g = 255;
            }


            // eros_thread.setCurrentTime(a.packetTime());
        }
        // eros_thread.setLatencyTime(input_port.getUnprocessedDelay());



//        if (pause)
//            m.lock();
//        else
//            m.unlock();

    }
}

double puckPosModule::getPeriod() {
    return 0.01;
}

bool puckPosModule::updateModule() {

//    cv::waitKey(1);
    char key = 0;
    m2.lock();

    cv::Mat eros_bgr, eros_filtered;
    cv::Mat eros_surface = EROS_vis.getSurface();
    cv::cvtColor(eros_surface, eros_bgr, cv::COLOR_GRAY2BGR);

    cv::Point puck_position = eros_thread.getState();
    cv::Rect puck_roi = eros_thread.getTrackROI();
    cv::Point  puck_init = eros_thread.getInitPos();

//    yInfo()<<puck_position.x<<" "<<puck_position.y;
//    cv::circle(eros_bgr, puck_position,5, cv::Scalar(180,119,31), cv::FILLED);
//    cv::rectangle(eros_bgr, cv::Point(puck_roi.x, puck_roi.y), cv::Point(puck_roi.x+puck_roi.width, puck_roi.y+puck_roi.height), cv::Scalar(180,119,31), 3);
//    if(eros_thread.getStatus())
//        cv::circle(eros_bgr, puck_init,5, cv::Scalar(255,0,0), cv::FILLED);
//    if(!eros_thread.getStatus())
//        cv::rectangle(eros_bgr, cv::Point(60, 150), cv::Point(560, 220), cv::Scalar(255,127,0), 3);

    cv::GaussianBlur(eros_bgr, eros_filtered, cv::Size(5, 5), 0);

    cv::imshow("FINAL TRACK", eros_filtered);
//    cv::waitKey(1);

    key = cv::waitKey(1);
    m2.unlock();

    if (key == 'p'){  // press p to pause
        pause = !pause;
        if(pause)
            m.try_lock();
        else
            m.unlock();
    }
    else if(key == 'n'){ // next
        m.unlock();
    }
    else if(key==32){
        eros_thread.setStatus(0);
    }

//    yInfo()<<"UPDATE MODULE";

    return Thread::isRunning();
}

bool puckPosModule::interruptModule() {
    eros_thread.stop();
    return Thread::stop();
}

void puckPosModule::onStop() {
    eros_thread.stop();
    input_port.stop();
}

void asynch_thread::initialise(cv::Mat &eros, int init_filter_width, cv::Rect roi, double thresh, std::mutex *m2, int n_trial, int n_exp)
{
    this->eros = eros; //shallow copy (i.e. pointer)
    this->m2=m2;

    detector.initialize(init_filter_width, roi, thresh); // create and visualize filter for detection phase
    tracker.initKalmanFilter();

    file_closed=true;

//    first_instant = yarp::os::Time::now();

//    file.open("/data/iros_datasets/exp"+std::to_string(n_exp)+"/Ours/Ours"+std::to_string(n_trial)+".txt");
////    file.open("/data/iros_datasets/live_test/puck"+std::to_string(n_trial)+".txt");
//
//    if (!file.is_open())
//    {
//        yError()<<"Could not open file for kalman prediction and correction";
//        return;
//    }

}

void asynch_thread::setStatus(int tracking_status) {
    this->tracking_status=tracking_status;
}

int asynch_thread::getStatus(){
    return tracking_status;
}

void asynch_thread::setFirstTime(double first) {
    this->startTime=first;
}

double asynch_thread::getFirstTime(){
    return startTime;
}
void asynch_thread::setCurrentTime(double current) {
    this->currentTime=current;
}
double asynch_thread::getCurrentTime(){
    return currentTime;
}

void asynch_thread::setLatencyTime(double latency) {
    this->latTime=latency;
}

double asynch_thread::getLatencyTime(){
    return latTime;
}

cv::Point asynch_thread::getState(){
    return tracker.getPosition();
}

cv::Rect asynch_thread::getTrackROI(){
    return tracker.get_convROI();
}

cv::Point asynch_thread::getInitPos(){
    return detector.getDetection();
}

void asynch_thread::run() {

    cv::Mat eros_filtered, kernel, result_visualization, temp;
    double tic = yarp::os::Time::now();
    double detection_time;
    bool first_detection=true;
    setStatus(0);

    while(!isStopping())
    {
        eros.copyTo(temp);
        cv::GaussianBlur(temp, eros_filtered, cv::Size(5, 5), 0);

        // --- DETECTION PHASE ----
        m2->lock();
        if (!getStatus())
        {
            if(detector.detect(eros_filtered)){
                setStatus(1);
                tracker.resetKalman(detector.getDetection(), detector.getSize());
                if(first_detection==true){
                    detection_time = getCurrentTime();
                    yInfo()<<"x="<<detector.getDetection().x<<",y="<<detector.getDetection().y<<",ts="<<detection_time-getFirstTime();
                    file<<detection_time-getFirstTime()<<" "<<detector.getDetection().x<<" "<<detector.getDetection().y<<" 0 0 0 0"<<endl;
                    first_detection=false;
                }
//                yInfo()<<"first detected = ("<<detector.getDetection().x<<","<<detector.getDetection().y<<")";
            }
        }
        // ---- TRACKING PHASE ----
        else{

            // UPDATE RATE
            double dT = yarp::os::Time::now() - tic;
            tic += dT;
            double eros_time_before= getCurrentTime();
//            yInfo() << "Running at a cool " << 1.0 / dT << "Hz";
            puck_pos = tracker.track(eros_filtered, dT);
            double eros_time_after = getCurrentTime();

            double eros_diff_time = eros_time_after-eros_time_before;
//            yInfo()<<latency;

//            yInfo()<<getLatencyTime();

            file<<(getCurrentTime()-getFirstTime())<<" "<<puck_pos.x<<" "<<puck_pos.y<<" "<<getLatencyTime()+eros_diff_time<<" "<<1.0 / dT<<" "<<tracker.get_convROI().width<<" "<<tracker.get_convROI().height<<endl;

//            if (puck_pos.x<=3){
//                setStatus(0);
//            }

//            if(detector.detect(eros_filtered)){
//                setStatus(1);
//                tracker.resetKalman(detector.getDetection(), detector.getSize());
//                detection_time = yarp::os::Time::now();
////                yInfo()<<"first detected = ("<<detector.getDetection().x<<","<<detector.getDetection().y<<")";
//            }

        }
        m2->unlock();
    }

}

int main(int argc, char *argv[]) {
    /* initialize yarp network */
    yarp::os::Network yarp;
    if (!yarp.checkNetwork()) {
        std::cout << "Could not connect to YARP" << std::endl;
        return false;
    }

    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setDefaultContext("event-driven");
//    rf.setDefaultConfigFile( "tracker.ini" );
    rf.setVerbose(false);
    rf.configure(argc, argv);

    /* create the module */
    puckPosModule puckpos;

    /* run the module: runModule() calls configure first and, if successful, it then runs */
    return puckpos.runModule(rf);
}