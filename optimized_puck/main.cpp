#include <yarp/os/all.h>
using namespace yarp::os;

#include "eros.h"
#include "tracking.h"
#include "visual_servoing.h"

class puckPos: public yarp::os::RFModule
{
    private:
        int w, h;
        double period{0.01};
        double eros_k, eros_d;
        cv::Point puck_pos;
        int init_filter_width, init_filter_height;
        double detection_thresh;
        cv::Rect detection_roi;

        tracking tracker;
        detection detector;
        std::thread computation_thread;
        std::ofstream fs;
        bool track_init{false}, detect_init{false};

        std::deque< std::array<double, 6> > data_to_save;

        EROSfromYARP eros_handler;
        headControl velocityController;

        bool active_control{false};

        struct fake_latency{
            cv::Point puck;
            double tstamp;
        };

        std::deque<fake_latency> fakeLat_queue;
        double tau{0.0};

    public:

        bool configure(yarp::os::ResourceFinder& rf) override
        {
            // options and parameters
            w = rf.check("w", Value(640)).asInt64();
            h = rf.check("h", Value(480)).asInt64();
            eros_k = rf.check("eros_k", Value(11)).asInt32();
            eros_d = rf.check("eros_d", Value(0.3)).asFloat64();
            period = rf.check("period", Value(0.01)).asFloat64();
            init_filter_width = rf.check("filter_w", Value(92)).asInt64();
            init_filter_height = rf.check("filter_h", Value(76)).asInt64();
            detection_thresh = rf.check("det_th", Value(30000)).asInt64();
            active_control = rf.check("control", Value(true)).asBool();

            // module name
            setName((rf.check("name", Value("/puck_position")).asString()).c_str());

            if (!eros_handler.start(cv::Size(w,h), "/zynqGrabber/AE:o", getName("/AE:i"), eros_k, eros_d)) {
                yError() << "could not open the YARP eros handler";
                return false;
            }

            cv::namedWindow("EROS", cv::WINDOW_AUTOSIZE);
            cv::resizeWindow("EROS", cv::Size(w,h));
            cv::moveWindow("EROS", 0,0);

            cv::namedWindow("RAW", cv::WINDOW_AUTOSIZE);
            cv::resizeWindow("RAW", cv::Size(w,h));
            cv::moveWindow("RAW", 0,540);

            cv::namedWindow("ROI_DETECT", cv::WINDOW_AUTOSIZE);
            cv::resizeWindow("ROI_DETECT", cv::Size(w,h));
            cv::moveWindow("ROI_DETECT", 700,0);

            cv::namedWindow("ROI_TRACK", cv::WINDOW_AUTOSIZE);
            cv::resizeWindow("ROI_TRACK", cv::Size(w,h));
            cv::moveWindow("ROI_TRACK", 700,540);

            if (rf.check("file")) {
                std::string filename = rf.find("file").asString();
                fs.open(filename);
                if (!fs.is_open()) {
                    yError() << "Could not open output file" << filename;
                    return false;
                }
            }

            if (active_control){
                if(!velocityController.initVelControl(h, w))
                    return false;

                if(!velocityController.initPosControl())
                    return false;

                velocityController.resetRobotHome();
            }

//            detection_roi = cv::Rect(60, 150, 500, 80);
            detection_roi = cv::Rect(270, 190, 100, 100);

            detect_init = detector.initialize(init_filter_width, init_filter_height, detection_roi, detection_thresh); // create and visualize filter for detection phase
            track_init = tracker.init();

            computation_thread = std::thread([this]{tracking_loop();});

            return true;
        }

        double getPeriod() override{
            return period;
        }

        bool updateModule() {

            static cv::Mat eros_bgr, eros_filtered;
            cv::cvtColor(eros_handler.eros.getSurface(), eros_bgr, cv::COLOR_GRAY2BGR);
            cv::GaussianBlur(eros_bgr, eros_filtered, cv::Size(5, 5), 0);
            cv::rectangle(eros_filtered, eros_handler.eros_update_roi, cv::Scalar(255,0,0), 1);
            cv::imshow("EROS", eros_filtered);
            cv::circle(eros_handler.raw_events, puck_pos,2, cv::Scalar(0,0,255), -1);
            cv::circle(eros_handler.raw_events, cv::Point(320, 240),2, cv::Scalar(0,255,0), -1);
            cv::imshow("RAW", eros_handler.raw_events);
            if (detect_init) {
                if(detector.strong_detection){
                    cv::circle(eros_filtered(detector.roi), detector.max_loc - cv::Point(detector.roi.x, detector.roi.y), 2, cv::Scalar(255, 0, 0), -1);
                    detect_init = false;
                }
                cv::imshow("ROI_DETECT", eros_filtered(detector.roi));
            }
            if (track_init)
                cv::imshow("ROI_TRACK", tracker.H);
            cv::waitKey(1);

            eros_handler.raw_events = 0;

            return true;
        }

        void tracking_loop() {

            cv::Mat eros_filtered;
            double tic = yarp::os::Time::now();
            bool tracking = false;
            double eros_diff_time;
            double errorTh = 3;
            double dT = 0;
            eros_handler.setEROSupdateROI(cv::Rect(0,0,640,480));

            yInfo()<<"tracking thread started";

            while (!isStopping()) {
                cv::GaussianBlur(eros_handler.eros.getSurface(), eros_filtered, cv::Size(5, 5), 0);

                if(tracking) {
                    dT = yarp::os::Time::now() - tic;
                    tic += dT;
//                    yInfo() << "Running at a cool " << 1.0 / dT << "Hz";
                    double eros_time_before = eros_handler.tic;
                    puck_pos = tracker.track(eros_filtered, dT);
                    eros_handler.setEROSupdateROI(tracker.roi);
                    fakeLat_queue.push_back({puck_pos, yarp::os::Time::now()});
//                    fakeLat_queue.push_back({puck_pos, yarp::os::Time::now()});
                    double eros_time_after = eros_handler.tic;
                    eros_diff_time = eros_time_after-eros_time_before;
                    if (fs.is_open() && eros_handler.tic > 0) {
                        data_to_save.push_back({eros_handler.tic, double(puck_pos.x), double(puck_pos.y), eros_handler.dt_not_read_events, eros_diff_time, dT});
                    }

                    if (active_control){

//                        yInfo()<<"NECK PITCH = "<<velocityController.getJointPos(0)<<", YAW = "<< velocityController.getJointPos(2);

                        tracker.yaw = velocityController.getJointPos(2);
                        static double trecord = yarp::os::Time::now();
                        double dt = yarp::os::Time::now() - trecord;
                        trecord += dt;

                        cv::Point sent_pos;
                        bool found_pos_sent=false;
                        while(fakeLat_queue.size()>0 && (yarp::os::Time::now()-fakeLat_queue.front().tstamp)>tau){
                            sent_pos = fakeLat_queue.front().puck;
                            found_pos_sent = true;
                            fakeLat_queue.pop_front();
                        }

                        double errorTh = 3; // pixels

//                        velocityController.closeToLimit(0);
//                        velocityController.closeToLimit(2);
                        if(found_pos_sent){
                            if (velocityController.computeErrorDistance(sent_pos.x, sent_pos.y) > errorTh){
                                velocityController.controlMono(sent_pos.x, sent_pos.y, dt);
//                            yInfo()<<"robot move";

                            }
                            else
                                velocityController.controlReset();
                        }

                    }
                }
                else {
//                    yInfo()<<"detection loop";
                    if (detector.detect(eros_filtered)) {
                        tracking = true;
                        yInfo()<<"detected puck"<<detector.max_loc.x<<detector.max_loc.y;
                        tracker.reset(detector.max_loc, detector.filter.cols, cv::Size(init_filter_width, init_filter_height));
                        if (fs.is_open() && eros_handler.tic > 0) {
                            data_to_save.push_back({eros_handler.tic, double(detector.max_loc.x), double(detector.max_loc.y), 0, 0, 0});
                        }
                    }
                }
            }
        }

        bool interruptModule() override {
            return true;
        }

        bool close() override {

            velocityController.controlReset();
            velocityController.resetRobotHome();

            yInfo() << "waiting for eros handler ... ";
            eros_handler.stop();
            yInfo() << "waiting for computation thread ... ";
            computation_thread.join();

            if(fs.is_open())
            {
                yInfo() << "Writing data";
                for(auto i : data_to_save)
                    fs << setprecision(2) << i[0] << " " << i[1] << " " << i[2] << " " << i[3] << " "<<i[4]<< " "<<i[5]<<" "<<i[6]<<" "<<i[7]<<" "<<i[8]<<" "<<i[9]<<std::endl;
                fs.close();
                yInfo() << "Finished Writing data";
            }

            return true;
        }
};

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
    puckPos puckpos;

    /* run the module: runModule() calls configure first and, if successful, it then runs */
    return puckpos.runModule(rf);
}