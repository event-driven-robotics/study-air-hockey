#pragma once

#include <yarp/os/all.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/math/Math.h>
#include <yarp/sig/all.h>

#include <event-driven/core.h>
#include "event-driven/algs.h"

#include <iostream>
#include <mutex>
#include <cmath>
#include <tuple>
#include <numeric>
#include <vector>
#include <deque>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>

#include "eros.h"

#define PI 3.14159265

using namespace ev;
using namespace cv;
using namespace yarp::os;
using namespace yarp::sig;
using namespace std;

// class detection
class detection {

private:

    double thresh;
    double width, height;
    map< pair<int, int>, cv::Mat> filter_set;

protected:

public:
    cv::Rect roi;
    cv::Point max_loc;
    cv::Mat filter;
    bool strong_detection{false};

    cv::Mat createEllipse(int puck_width, int puck_height){

        width = puck_width;
        height = puck_height;
        cv::Point2d origin((width)/2, (height)/2);

        cv::Mat ell_filter = cv::Mat::zeros(height, width, CV_32F);

        for(int x=0; x< ell_filter.cols; x++) {
            for(int y=0; y< ell_filter.rows; y++) {
                double dx = (pow(x,2) -2*origin.x*x + pow(origin.x,2))/pow((width)/2,2);
                double dy = (pow(y,2) -2*origin.y*y + pow(origin.y,2))/pow((height)/2,2);
                double value = dx+ dy;
                if(value > 1)
                    ell_filter.at<float>(y, x) = 0;
                else if (value > 0.6 && value<=1)
                    ell_filter.at<float>(y, x) = 1;
                else
                    ell_filter.at<float>(y, x) = 0;

            }
        }
        return ell_filter;
    }

    bool initialize(int filter_width, int filter_height, cv::Rect roi, double thresh){

        this -> roi = roi;
        this -> thresh = thresh;

        filter = createEllipse(filter_width, filter_height);

        return true;
    }

    bool detect(cv::Mat eros){

        static cv::Mat surface, result_convolution, result_visualization, result_color, result_conv_normalized, heat_map, result_final;
        double min, max; cv::Point min_loc;

        eros(roi).convertTo(surface, CV_32F);

        cv::filter2D(surface, result_convolution, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_ISOLATED); // look at border
        cv::minMaxLoc(result_convolution, &min, &max, &min_loc, &max_loc);

//        cv::normalize(result_convolution, result_conv_normalized, 255, 0, cv::NORM_MINMAX);
//        result_conv_normalized.convertTo(result_visualization, CV_8U);

//        cv::cvtColor(result_visualization, result_color, cv::COLOR_GRAY2BGR);
//        if (max>thresh)
//            cv::circle(result_color, max_loc, 5, cv::Scalar(255, 0, 0), cv::FILLED);
//        else
//            cv::circle(result_color, max_loc, 5, cv::Scalar(0, 0, 255), cv::FILLED);

//        cv::imshow("DETECT_MAP", eros(roi));

//        result_convolution.at<float>(0,0) = 4000;
//        cv::normalize(result_convolution, heat_map, 0, 255, cv::NORM_MINMAX);
//        heat_map.convertTo(heat_map, CV_8U);
//
//        cv::applyColorMap(heat_map, result_final, cv::COLORMAP_JET);

//        cv::imshow("DETECT_HEAT_MAP", result_final);
//        cv::waitKey(1);
//        yInfo()<<max;

        max_loc += cv::Point(roi.x, roi.y);

        strong_detection = max>thresh;

        return strong_detection;
    }
};

// class tracking
class tracking{

private:

    double factor; // should be positive (roi width > puck size) and fixed
    int puck_size;
    map<int, cv::Mat> filter_bank;
    cv::Rect roi_full;
    map< pair<int, int>, cv::Mat> filter_set;
    map< int, cv::Mat> filter_yaws;
    int filter_bank_min, filter_bank_max;
    cv::Point2d puck_meas;
    typedef struct{cv::Point p; double s;} score_point;
    score_point best;
    double first_time;
    cv::Point starting_position;
    cv::Size init_filter_size;
    cv::Rect around_puck;

    void createFilterBank(int min, int max){
        for(int i=min; i<=max; i+=2){
            for(int j=min; j<=max; j+=2){
                filter_set[make_pair(i,j)] = createCustom(i, j);
            }
        }
    }

    void createFilterBankYaw(){

        int yaw_min = -45;
        int yaw_max = 45;

        cv::Size filter_dim;
        double filter_theta;

        for(int i=yaw_min; i<=yaw_max; i+=1){
            std::tie(filter_dim, filter_theta) = kernel_target_following(double(i));
//            yInfo()<<i<<filter_dim.width << filter_dim.height << filter_theta;
            filter_yaws[i] = createOrientedEllipse(filter_dim.width+19, filter_dim.height+19, 360-filter_theta);
//            imshow("filter", filter_yaws[i]);
//            waitKey(1000);
        }
    }

    score_point convolution(cv::Mat eros, cv::Mat filter, double x_puck_pos, double y_puck_pos) {

        static cv::Mat surface, result_convolution, result_visualization, result_surface, result_color, result_final, result_final_filtered, result_conv_normalized, heat_map;
        double min, max;
        cv::Point highest_peak_filtered, lowest_peak, highest_peak;

        roi = roi & roi_full;

        eros(roi).convertTo(surface, CV_32F);
        cv::filter2D(surface, result_convolution, -1, filter, cv::Point(-1, -1), 0,
                     cv::BORDER_ISOLATED); // look at border
        cv::Rect zoom = Rect(x_puck_pos - filter.cols * 0.5 - roi.x, y_puck_pos - filter.rows * 0.5 - roi.y, filter.cols+6,
                     filter.rows+6) & cv::Rect (0,0,result_convolution.cols, result_convolution.rows);
        cv::minMaxLoc(result_convolution(zoom), &min, &max, &lowest_peak, &highest_peak);
        cv::normalize(surface, result_surface, 255, 0, cv::NORM_MINMAX);
        result_surface.convertTo(result_visualization, CV_8U);

        cv::cvtColor(result_visualization, result_color, cv::COLOR_GRAY2BGR);

        cv::normalize(result_convolution, result_conv_normalized, 0, 255, cv::NORM_MINMAX);

        result_conv_normalized.convertTo(heat_map, CV_8U);

        Mat g = getGaussianKernel(zoom.height, 0.5* filter.rows, CV_32F) *
                getGaussianKernel(zoom.width, 0.5* filter.cols, CV_32F).t();

        Mat heat_map_zoom = result_conv_normalized(zoom);
        Mat heat_map_filtered = heat_map_zoom.mul(g);

        cv::normalize(heat_map_filtered, heat_map_filtered, 0, 255, cv::NORM_MINMAX);
        heat_map_filtered.convertTo(heat_map_filtered, CV_8U);

        cv::applyColorMap(heat_map, result_final, cv::COLORMAP_JET);
        cv::applyColorMap(heat_map_filtered, result_final_filtered, cv::COLORMAP_JET);

        hconcat(result_color, result_final, H);

        cv::minMaxLoc(heat_map_filtered, &min, &max, &lowest_peak, &highest_peak_filtered);
        cv::Point new_peak = highest_peak + cv::Point(zoom.x, zoom.y);
        cv::Point new_peak_filtered = highest_peak_filtered + cv::Point(zoom.x, zoom.y);

//        cv::circle(H, new_peak, 2, cv::Scalar(0, 0, 255), cv::FILLED);
//        cv::circle(H, prev_peak-cv::Point(roi.x,roi.y), 2, cv::Scalar(0, 255, 0), cv::FILLED);
//        prev_peak = new_peak+cv::Point(roi.x, roi.y);
//        cv::circle(H, new_peak_filtered, 2, cv::Scalar(0, 0, 255), cv::FILLED);
//        cv::circle(H, weighted_pos, 2, cv::Scalar(255, 0, 0), cv::FILLED);
        cv::rectangle(H, zoom, cv::Scalar(0, 255, 0));
//        cv::ellipse(H, cv::Point(zoom.x+filter.cols*0.5, zoom.y+filter.rows*0.5), Size(filter.cols*0.5, filter.rows*0.5), 0,0,360,cv::Scalar(0,0,255),1);

//        yInfo()<<"width = "<<filter.cols<<", height = "<<filter.rows;

        cv::Rect zoom2 = cv::Rect(zoom.x+result_color.cols, zoom.y, zoom.width, zoom.height);
        cv::rectangle(H, zoom, cv::Scalar(255, 0, 255));
        cv::rectangle(H, zoom2, cv::Scalar(255, 0, 255));

//        cv::imshow("ROI TRACK", H);
//        cv::imshow("ZOOM", result_final(zoom));
//        cv::imshow("GAUSSIAN MUL", result_final_filtered);

//        cv::waitKey(1);

        return {new_peak_filtered + cv::Point(roi.x, roi.y), max};
    }

    cv::Point multi_conv(cv::Mat eros, int width, int height){

        auto p = convolution(eros, filter_set[make_pair(width,height)], puck_meas.x, puck_meas.y);

//        for (int i=0;i<filter_set[make_pair(width,height)].rows; i++){
//            for (int j=0;j<filter_set[make_pair(width,height)].cols; j++){
//
//                std::cout<<filter_set[make_pair(width,height)].at<float>(i,j)<<" ";
//            }
//            std::cout<<std::endl;
//        }
//        cv::Mat dog_filter_grey;
//        cv::normalize(filter_set[make_pair(width,height)], dog_filter_grey, 0, 255, cv::NORM_MINMAX);
//        cv::Mat vis_ellipse, color_ellipse;
//        dog_filter_grey.convertTo(vis_ellipse, CV_8U);
//        cv::cvtColor(vis_ellipse,color_ellipse, cv::COLOR_GRAY2BGR);
//        cv::imshow("ell_filter", color_ellipse);

//        cv::imshow("ell_filter",filter_set[make_pair(width,height)]);
//        cv::waitKey(1);

        if(p.s > 0){
            best = p;
        }

        return best.p;
    }

    cv::Point multi_conv_yaw(cv::Mat eros, double yaw){

        if (yaw>45)
            yaw=45;
        else if (yaw<-45)
            yaw=-45;
//        yInfo()<<yaw<<filter_yaws[int(yaw)].cols<<filter_yaws[int(yaw)].rows;
        auto p = convolution(eros, filter_yaws[int(yaw)], puck_meas.x, puck_meas.y);

//        for (int i=0;i<filter_set[make_pair(width,height)].rows; i++){
//            for (int j=0;j<filter_set[make_pair(width,height)].cols; j++){
//
//                std::cout<<filter_set[make_pair(width,height)].at<float>(i,j)<<" ";
//            }
//            std::cout<<std::endl;
//        }
//        cv::Mat dog_filter_grey;
//        cv::normalize(filter_set[make_pair(width,height)], dog_filter_grey, 0, 255, cv::NORM_MINMAX);
//        cv::Mat vis_ellipse, color_ellipse;
//        dog_filter_grey.convertTo(vis_ellipse, CV_8U);
//        cv::cvtColor(vis_ellipse,color_ellipse, cv::COLOR_GRAY2BGR);
//        cv::imshow("ell_filter", color_ellipse);

//        cv::imshow("ell_filter",filter_set[make_pair(width,height)]);
//        cv::imshow("ell_filter",filter_yaws[int(yaw)]);
//        cv::waitKey(1);

        if(p.s > 0){
            best = p;
        }

        return best.p;
    }

protected:

public:

    cv::Mat H;
    double yaw;
    cv::Rect roi;

    struct fake_latency{
        cv::Point puck;
        double tstamp;
    };

    std::deque<fake_latency> fakeLat_queue;

    bool init(){

        H = cv::Mat::zeros(480, 640, CV_32F);

        factor = 2;
        roi_full = cv::Rect(0,0,640,480);

//        filter_bank_min = 29;
//        filter_bank_max = 79;
//        createFilterBank(filter_bank_min, filter_bank_max);
        createFilterBankYaw();

//        cv::Mat oriented_ellipse_mat = createOrientedEllipse(55,47,45);
//
//        cv::imshow("oriented_ellipse",oriented_ellipse_mat);

        first_time = yarp::os::Time::now();

        return(true);
    }

    void updateROI(Point2d position, int width, int height){

        float u = position.x;
        float v = position.y;

        //puck_size = 0.1*v;
        int roi_width = factor*width;
        int roi_height = factor*height;

        if(roi_width%2==0)
            roi_width++;

        if(roi_height%2==0)
            roi_height++;

        cv::Rect roi_full = cv::Rect(0,0,640,480);
        roi = cv::Rect(u - roi_width/2, v - roi_height/2, roi_width, roi_height) & roi_full;
//        yInfo()<<"ROI ="<<roi.x<<" "<<roi.y<<" "<<roi.width<<" "<<roi.height;
    }

    void reset(cv::Point starting_position, int puck_size, cv::Size filter_size){

        score_point best={starting_position, 5000.0};

        if (puck_size%2 == 0)
            puck_size++;

        this->puck_size=puck_size;
        this->starting_position=starting_position;
        this->init_filter_size= filter_size;

        updateROI(starting_position, filter_size.width, filter_size.height);

        puck_meas = starting_position;
        roi = cv::Rect(starting_position.x-puck_size/2, starting_position.y-puck_size/2, filter_size.width, filter_size.height);
    }

    cv::Mat createCustom(int width, int height){

        cv::Point2d origin((width+4)/2, (height+4)/2);
        cv::Mat ell_filter = cv::Mat::zeros(height+4, width+4, CV_32F);

//        std::cout<<ell_filter.cols<<" "<<ell_filter.rows;

        for(int x=0; x< ell_filter.cols; x++) {
            for(int y=0; y< ell_filter.rows; y++) {
                double dx = (pow(x,2) -2*origin.x*x + pow(origin.x,2))/pow((width)/2,2);
                double dy = (pow(y,2) -2*origin.y*y + pow(origin.y,2))/pow((height)/2,2);
                double value = dx+ dy;
                if(value > 0.8)
                    ell_filter.at<float>(y, x) = -0.2;
                else if (value > 0.5 && value<=0.8)
                    ell_filter.at<float>(y, x) = 3;
                else
                    ell_filter.at<float>(y, x) = -1.0;

//                std::cout<<ell_filter.at<float>(y,x)<<" ";

            }
//            std::cout<<std::endl;
        }
        return ell_filter;
    }

    cv::Mat createOrientedEllipse(int width, int height, double theta){

        cv::Point2d origin((width+20)/2, (height+40)/2);
        cv::Mat ell_filter = cv::Mat::zeros(height+40, width+20, CV_32F);

        double theta_rad = theta*PI/180.0;
//        std::cout<<ell_filter.cols<<" "<<ell_filter.rows;

        for(int x=0; x< ell_filter.cols; x++) {
            for(int y=0; y< ell_filter.rows; y++) {

                double factor1 = (x-origin.x)*cos(theta_rad);
                double factor2 = (y-origin.y)*sin(theta_rad);
                double factor3 = (x-origin.x)*sin(theta_rad);
                double factor4 = (y-origin.y)*cos(theta_rad);
                double dx = (pow(factor1,2) + 2*factor1*factor2 + pow(factor2,2))/pow((width)/2,2);
                double dy = (pow(factor3,2) - 2*factor3*factor4 + pow(factor4,2))/pow((height)/2,2);
                double value = dx+ dy;
                if(value > 0.8)
                    ell_filter.at<float>(y, x) = -0.1;
                else if (value > 0.6 && value<=0.8)
                    ell_filter.at<float>(y, x) = 3;
                else
                    ell_filter.at<float>(y, x) = -0.3;

//                std::cout<<ell_filter.at<float>(y,x)<<" ";

            }
//            std::cout<<std::endl;
        }
        return ell_filter;
    }

    cv::Size changing_kernel_size_air_hockey(cv::Point previous_puck_pos, int experiment){
        double a0,a1,a2,b0,b1,b2;
        if (experiment==1){
            // static scenario params
            a0 = 17.715624038044254 , a1 = 0.0035291125887034315 , a2 = 0.09579542600737656;
            b0 = 6.313115087846054 , b1 = 0.0022264258939534796 , b2 = 0.09551609558936817;
        }
        else{
            a0 = 12.201650881598578 , a1 = 0.009735421154783392 , a2 = 0.09872367924641742;
            b0 = 1.92477315662196 , b1 = 0.003209378102266541 , b2 = 0.09801072879753021;
        }

        double width = a0+a1*previous_puck_pos.x+a2*previous_puck_pos.y;
        double height = b0+b1*previous_puck_pos.x+b2*previous_puck_pos.y;
        height = 2 * floor(height/2) + 1;
        width = 2 * floor(width/2) + 1;

        if(height<9)
            height = 9;
        if(width<9)
            width = 9;

        return cv::Size(width, height);
    }

    std::tuple<cv::Size, double> kernel_target_following(double yaw){

        double k0_width = 76.3925, k1_width = 0.0921, k2_width = -0.0147;
        double k0_height = 59.8906, k1_height = 0.267, k2_height = -0.0133;
        double m_theta, q_theta;
        if (yaw>0){
            m_theta = 1.244;
            q_theta = -7.706;
        }
        else{
            m_theta = 1.044;
            q_theta = 353.9;
        }
        double width = k0_width + k1_width*yaw + k2_width*yaw*yaw;
        double height = k0_height + k1_height*yaw + k2_height*yaw*yaw;
        double theta_deg = m_theta*yaw + q_theta;
//        yInfo()<<"theta"<<yaw<<m_theta<<q_theta<<theta_deg;
        height = 2 * floor(height/2) + 1;
        width = 2 * floor(width/2) + 1;

        if(height<9)
            height = 9;
        if(width<9)
            width = 9;

        cv::Size axes_ellipse = cv::Size(width, height);

        return std::make_tuple(axes_ellipse, theta_deg);
    }

    cv::Point track(cv::Mat eros, double dT){

        static cv::Mat eros_bgr;
        cv::Size kernel_size;
        double ellipse_orient;
//        kernel_size = changing_kernel_size_air_hockey(puck_meas, 1);
        std::tie(kernel_size, ellipse_orient) = kernel_target_following(yaw);

        double width = kernel_size.width+19;
        double height = kernel_size.height+19;

//        double width = init_filter_size.width;
//        double height = init_filter_size.height;

//        puck_meas = multi_conv(eros, width, height);
        puck_meas = multi_conv_yaw(eros, yaw);

        around_puck = cv::Rect(puck_meas.x - width/2, puck_meas.y - height/2, width, height) & roi_full;
        updateROI(puck_meas, width, height);

        return puck_meas;
    }

};
