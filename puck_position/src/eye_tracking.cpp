#include "eye_tracking.h"

int main(int argc, char *argv[])
{
    /* initialize yarp network */
    yarp::os::Network yarp;
    if (!yarp.checkNetwork())
    {
        std::cout << "Could not connect to YARP" << std::endl;
        return false;
    }

    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setDefaultContext("event-driven");
    //  rf.setDefaultConfigFile( "tracker.ini" );
    rf.setVerbose(false);
    rf.configure(argc, argv);

    /* create the module */
    eyeTracker puckpos;

    /* run the module: runModule() calls configure first and, if successful, it then runs */
    return puckpos.runModule(rf);
}

cv::Mat eyeTracker::createEllipse(int width, int height)
{

    cv::Point2d origin((width) / 2, (height) / 2);
    cv::Mat ell_filter = cv::Mat::zeros(height, width, CV_32F);

    for (int x = 0; x < ell_filter.cols; x++)
    {
        for (int y = 0; y < ell_filter.rows; y++)
        {
            double dx = (pow(x, 2) - 2 * origin.x * x + pow(origin.x, 2)) / pow((width) / 2, 2);
            double dy = (pow(y, 2) - 2 * origin.y * y + pow(origin.y, 2)) / pow((height) / 2, 2);
            double value = dx + dy;
            std::cout << value<<endl;
            if (value > 0.7)//0.95
                ell_filter.at<float>(y, x) = 0;
            else if (value > 0.5 && value <= 0.7)//>0.6 <0.9
                ell_filter.at<float>(y, x) = 1;
            else
                ell_filter.at<float>(y, x) = 0;

            //std::cout << ell_filter.at<float>(y,x);
        }
    }

    
    return ell_filter;

}
   



void eyeTracker::make_template(const cv::Mat &input, cv::Mat &output)
{
    static cv::Mat pos_hat, neg_hat;
    double blur{9};
    static cv::Size pblur(blur, blur);
    static cv::Size nblur(2 * blur - 1, 2 * blur - 1);
    static double minval, maxval;

    cv::GaussianBlur(input, pos_hat, pblur, 0);
    cv::GaussianBlur(input, neg_hat, nblur, 0);
    output = pos_hat - neg_hat;
    cv::minMaxLoc(output, &minval, &maxval);
    double scale_factor = 1.0 / (2 * std::max(fabs(minval), fabs(maxval)));
    output *= scale_factor;
}




void eyeTracker::normalized_filter( cv::Mat &out_mexican)
{
    double total_p, scalar;
    double max_value;
    double max= 0.00;

    cv::Mat pfilter = cv::max(out_mexican,0);
    total_p = cv::sum(pfilter)[0];
    scalar = 100.0/total_p;
    //yInfo() << scalar;
    cv::minMaxLoc( out_mexican, nullptr, &max_value);

    //yInfo() <<max_value;

    out_mexican *= scalar;
    cv::minMaxLoc( out_mexican, nullptr, &max_value);
    //yInfo() <<max_value;

}


void eyeTracker::run_eros()
{

    while (!isStopping())
    {
        my_info = input_port.readAll(true);

        for (auto &v : input_port)
        {
            eros.update(v.x, v.y);
            raw_events.at<cv::Vec3b>(v.y, v.x) = cv::Vec3b(255, 255, 255);
        }
    
    }
}

void eyeTracker::run_detector()
{
    cv::Mat eros_filtered, surface;
    double detection_time = 0;


    while (!isStopping())
    {   
        cv::GaussianBlur(eros.getSurface(), eros_filtered, cv::Size(5, 5), 0);
        eros_filtered.convertTo(surface, CV_32F);


        cv::filter2D(surface, result_surface_det, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_ISOLATED); // look at border
        cv::minMaxLoc(result_surface_det, nullptr, &detector_score, nullptr, &detector_position);
   
        //yInfo() <<detector_score;

        detection_time=  my_info.timestamp;
        //yInfo() << detection_time;
        //std::cout << "x="<<detector_position.x<<" " <<"y="<<detector_position.y<< " " <<"ts="<<detection_time <<std::endl;
        if (detector_score > threshold)
            new_detection = true;

        if(detection_time > 0.1e-3)
            file<<detection_time<<" " <<detector_position.x <<" " <<detector_position.y<< " " <<detector_score <<endl;

    }
}


void eyeTracker::run_tracker()
{
   
    cv::Mat eros_filtered, heat_map, output_surface;
    cv::Mat convolution_norm, convolution;
    cv::Mat norm_conv, result_final_filtered, result_color, result_visualization;
    double tracking_time;
    
   
    while (!isStopping())
    {
        
        double gt_x = 176.5;
        double gt_y = 103.5;
        double factor = 1.1;
        int roi_w = factor*filter_w;
        int roi_h = factor*filter_h;
       
        
        cv::GaussianBlur(eros.getSurface(), eros_filtered, cv::Size(5, 5), 0);
        eros_filtered.convertTo(surface_track, CV_32F);
     
        cv::filter2D(surface_track, convolution, -1, filter, cv::Point(-1,-1), 0, cv::BORDER_ISOLATED);

        if (tracker_position_filtered.x == 0 && tracker_position_filtered.y == 0)

            roi = Rect( gt_x- roi_w*0.5, gt_y - roi_h*0.5, roi_w, roi_h)
                        & Rect(0, 0, w, h);

        else

            roi = Rect(peak.x - roi_w*0.5, peak.y - roi_h*0.5, roi_w, roi_h)
                    &Rect(0, 0, w, h);
        

        cv::minMaxLoc(convolution(roi), nullptr, &tracker_score, nullptr, &tracker_position);

        // yInfo()<< " tracker position" << tracker_position.x;
        cv::normalize(convolution, convolution_norm, 0, 255, cv::NORM_MINMAX);
            convolution_norm.convertTo(heat_map, CV_8U);

        Mat g = getGaussianKernel( roi.height, 0.5*filter.rows, CV_32F) *
                getGaussianKernel( roi.width, 0.5*filter.cols, CV_32F).t();

        Mat heat_map_roi= convolution_norm(roi);
        conv = heat_map_roi.mul(g);
        
        cv::minMaxLoc(conv, nullptr, &tracker_score, nullptr, &tracker_position_filtered);

        // yInfo() << "tracker position filtered" << tracker_position_filtered.x;


        peak =  tracker_position_filtered + cv::Point(roi.x, roi.y);

        tracking_time= my_info.timestamp;

        if(tracking_time > 0.1e-3)

        file2<<tracking_time<< " " << peak.x << " " << peak.y<<endl;
    
        cv::normalize(conv, norm_conv, 0, 255, cv::NORM_MINMAX);
            norm_conv.convertTo(norm_conv, CV_8U);

        cv::applyColorMap(norm_conv, result_final_filtered, cv::COLORMAP_JET);

        // cv::circle(result_color, tracker_position, 3, cv::Scalar(0,255,255), -1,8);

        cv::imshow("GAUSSIAN MUL", result_final_filtered);
     
    }
}

bool eyeTracker::configure(yarp::os::ResourceFinder &rf)
{

    //parameters
    w = rf.check("w", Value(346)).asInt();
    h = rf.check("h", Value(260)).asInt();

    filter_w = rf.check("filter_w", Value(65)).asInt();
    filter_h = rf.check("filter_h", Value(60)).asInt();


    // yarp stuff
    setName((rf.check("name", Value("/eye-tracking")).asString()).c_str());

    if (!input_port.open(getName() + "/AE:i"))
        return false;

    raw_events = cv::Mat::zeros(h,w, CV_8UC3);
    filter = createEllipse(filter_w, filter_h);
    make_template(filter, filter);
    normalized_filter(filter);

    cv::imshow("my filter", filter+0.5);


    eros.init(w, h, 9, 0.2);

    cv::namedWindow("DETECT_HEAT_MAP", cv::WINDOW_NORMAL);
    cv::moveWindow("DETECT_HEAT_MAP", 500,500);

    cv::namedWindow("EVENTS", cv::WINDOW_NORMAL);
    cv::moveWindow("EVENTS", 600,600);

    cv::namedWindow("ROI TRACK", cv::WINDOW_NORMAL);
    cv::moveWindow("ROI TRACK", 600,600);

    eros_thread = std::thread([this]{run_eros();});
    detector_thread = std::thread([this]{run_detector();});
    tracker_thread = std::thread([this]{run_tracker();});

    yarp::os::Network::connect("/file/ch0dvs:o", getName("/AE:i"), "fast_tcp");

    file.open("/data/final_dataset_pc_roberta/Solo_detection/New_users2/User6_0.txt");
    file2.open("/data/final_dataset_pc_roberta/Solo_tracking/User6_0.txt");
   
    if(!file.is_open())
    {
        yError()<<"Could not open file detection";
        return false;
    }

    if (!file2.is_open())
    {
        yError()<<"Could not open file tracking";
        return false;
    }

    return true;
}

bool eyeTracker::interruptModule()
{
    input_port.stop();
    return true;
}
double eyeTracker::getPeriod()
{
    return 0.033;
}
bool eyeTracker::updateModule()
{ 
    //visualisation thread
    static cv::Mat result_surface_norm;
    static cv::Mat heat_map, result_final, overlay, overlay_color, filter_norm, output_mat;
    static cv::Mat output_surface, result_color,result_visualization;
    static cv::Mat result_final_filtered, norm_conv, filter_color;

    //raw_events = 0;


    cv::normalize(surface_track, overlay, 255, 0, cv::NORM_MINMAX);
    overlay.convertTo(overlay, CV_8U);

    std::vector<cv::Mat> channels; 
    channels.push_back(cv::Mat::zeros(h, w, CV_8U));
    channels.push_back(overlay);
    channels.push_back(cv::Mat::zeros(h, w, CV_8U));
    cv::merge(channels, overlay_color);

    cv::normalize(filter, filter_norm, 255, 0, cv::NORM_MINMAX);
        filter_norm.convertTo(filter_norm, CV_8U);

    cv::cvtColor(filter_norm, filter_color, cv::COLOR_GRAY2BGR);


    cv::Rect roi_nf= Rect(detector_position.x-0.5*filter_w, detector_position.y-0.5*filter_h, filter_w, filter_h);
    cv::Rect roi_n= roi_nf & cv::Rect(0, 0, w, h);
    
    if(roi_n == roi_nf)
        overlay_color(roi_n) = overlay_color(roi_n) + filter_color*0.3;

    cv::imshow("overlay", overlay_color);


    /// DETECTION ///

    cv::imshow("EVENTS", raw_events);

    raw_events = 0;

    cv::normalize(result_surface_det, result_surface_norm, 0.0, 1.0, cv::NORM_MINMAX);

    cv::normalize(result_surface_det, heat_map, 0, 255, cv::NORM_MINMAX);
    heat_map.convertTo(heat_map, CV_8U);

    cv::applyColorMap(heat_map, result_final, cv::COLORMAP_JET);

    if (detector_score> threshold)
    cv::circle(raw_events, detector_position, 5, cv::Scalar(0, 0, 255), cv::FILLED);
    else
    cv::circle(raw_events, detector_position, 5, cv::Scalar(0, 0, 0), cv::FILLED);

    cv::imshow("DETECT_HEAT_MAP", result_final);  
    
    /// TRACKING ///

    cv::normalize(surface_track, output_surface, 255, 0, cv::NORM_MINMAX);
        output_surface.convertTo(result_visualization, CV_8U);

    cv::normalize(conv, norm_conv, 0, 255, cv::NORM_MINMAX);
        norm_conv.convertTo(norm_conv, CV_8U);

    cv::cvtColor(result_visualization, result_color, cv::COLOR_GRAY2BGR);

    //cv::applyColorMap(norm_conv, result_final_filtered, cv::COLORMAP_JET);

    cv::rectangle(result_color, roi, cv::Scalar(0, 255, 0));
    cv::circle(result_color, peak, 3, cv::Scalar(255, 0, 255), -1, 8);
    cv::circle(result_color, tracker_position_filtered, 3, cv::Scalar(255, 0, 255), -1, 8);

    cv::imshow("ROI TRACK", result_color);
    //cv::imshow("GAUSSIAN MUL", result_final_filtered);

    

    cv::waitKey(1);

    return true;

}