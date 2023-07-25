
#pragma once

#include <yarp/os/all.h>
#include <event-driven/core.h>
#include <event-driven/algs.h>
#include <thread>

#define PI 3.14159265
using namespace ev;
using namespace cv;
using namespace yarp::os;
using namespace std;




class eyeTracker : public RFModule {

private:

    //variables
    ofstream file;
    ofstream file2;
    ev::EROS eros;
    ev::info my_info;
    ev::window<ev::AE> input_port;
    int w{640}, h{480};
    int filter_w, filter_h;

    cv::Point detector_position{-1, -1};
    cv::Point tracker_position{-1, -1};
    cv::Point tracker_position_filtered{-1, -1};
    cv::Point peak;
    cv::Rect roi;
    double    detector_score{0};
    double    tracker_score{0};
    double    threshold=1200;
    
    bool new_detection = false;

    //threads
    std::thread eros_thread;
    std::thread detector_thread;
    std::thread tracker_thread;


    //images
    cv::Mat filter;
    cv::Mat raw_events;
    cv::Mat result_surface_det;
    cv::Mat surface_track;
    cv::Mat conv;


    void run_eros();
    void run_detector();
    void run_tracker();

    cv::Mat createEllipse(int width, int height);
    void make_template(const cv::Mat &input, cv::Mat &output);
    void normalized_filter(cv::Mat &out_mexican);

public:

    // the virtual functions that need to be overloaded
    bool configure(yarp::os::ResourceFinder &rf) override;
    bool interruptModule() override;
    double getPeriod() override;
    bool updateModule() override;

    

};