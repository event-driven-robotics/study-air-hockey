#pragma once

#include <yarp/os/all.h>
#include <event-driven/algs.h>
#include <event-driven/vis.h>
#include "tracking.h"

using namespace yarp::os;

class EROSfromYARP
{
public:

    ev::window<ev::AE> input_port;
    ev::EROS eros;
    std::thread eros_worker;
    double tic{-1};
    double dt_not_read_events{0};
    int num_not_read_events{0};
    cv::Mat raw_events;
    cv::Rect eros_update_roi;
    double tau{0.03};

    void setEROSupdateROI(cv::Rect roi) {
        this -> eros_update_roi = roi;
    }

    void erosUpdate() 
    {
        while (!input_port.isStopping()) {
            ev::info my_info = input_port.readAll(true);
//            yarp::os::Time::delay(tau);
//            tic = my_info.timestamp;
            for(auto &v : input_port){
//                yInfo()<<eros_update_roi.x<<eros_update_roi.y<<eros_update_roi.width<<eros_update_roi.height;
                if((640-v.x) > eros_update_roi.x && (640-v.x) < eros_update_roi.x + eros_update_roi.width && (480-v.y) > eros_update_roi.y && (480-v.y) < eros_update_roi.y + eros_update_roi.height)
                    eros.update(640-v.x, 480-v.y);
                raw_events.at<cv::Vec3b>(480-v.y, 640-v.x) = cv::Vec3b(255,255, 255);
            }

//            dt_not_read_events = input_port.stats_unprocessed().duration;
//            num_not_read_events = input_port.stats_unprocessed().count;
        }
    }

public:
    bool start(cv::Size resolution, std::string sourcename, std::string portname, int k = 5, double d = 0.3)
    {
        eros.init(resolution.width, resolution.height, k, d);

        if (!input_port.open(portname))
            return false;

        raw_events = cv::Mat::zeros(480,640, CV_8UC3);

        yarp::os::Network::connect(sourcename, portname, "fast_tcp");

        eros_worker = std::thread([this]{erosUpdate();});
        return true;
    }

    void stop()
    {
        input_port.stop();
        eros_worker.join();
    }

};
