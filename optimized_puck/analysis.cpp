#include <yarp/os/all.h>
using namespace yarp::os;

#include "eros.h"
#include "tracking.h"

class objectPos: public yarp::os::RFModule
{
private:
    int w, h;
    double period{0.1};

    ev::window<ev::AE> input_port;

    ev::EROS eros;
    int eros_k;
    double eros_d;
    cv::Mat eros_filtered, eros_detected, eros_detected_roi, eros_tracked, eros_tracked_roi;

    detection detector;
    int init_filter_width, init_filter_height;
    double detection_thresh;
    cv::Rect detection_roi;
    cv::Rect roi_full;

    tracking tracker;
    bool tracking_bool;
    cv::Point object_position;

    std::thread affine_thread;
    std::ofstream fs;

    typedef struct affine_struct
    {
        cv::Mat A;
        cv::Mat warped_img, warped_img_color;
        double score;

    } affine_bundle;

    std::array<affine_bundle, 7> affine_info;
    double translation{1}, angle{0.5}, pscale{1.5}, nscale{0.5};
    std::vector<cv::Mat> affines_vector;
    std::vector<double> scores_vector;
    cv::Mat last_template, new_template;
    cv::Mat last_position_homogenous, new_position;
    cv::Rect last_roi, new_roi;

    cv::Mat black_image;

    double similarity_score(const cv::Mat &observation, const cv::Mat &expectation) {
        static cv::Mat muld;
        muld = expectation.mul(observation);
        return cv::sum(cv::sum(muld))[0];
    }

    void printMatrix(Mat &matrix) const {
        for (int r=0; r < matrix.rows; r++){
            for (int c=0; c<matrix.cols; c++){
                cout<<matrix.at<double>(r,c)<<" ";
            }
            cout << endl;
        }
    }

    void initializeAffines(){
        for(auto &affine : affine_info) {
            affine.A = cv::Mat::zeros(2,3, CV_64F);
        }
    }

    void createAffines(double translation, cv::Point2f center, double angle, double pscale, double nscale){
        initializeAffines();

        affine_info[0].A.at<double>(0,0) = 1;
        affine_info[0].A.at<double>(0,2) = translation;
        affine_info[0].A.at<double>(1,1) = 1;
        affine_info[0].A.at<double>(1,2) = 0;

        affine_info[1].A.at<double>(0,0) = 1;
        affine_info[1].A.at<double>(0,2) = -translation;
        affine_info[1].A.at<double>(1,1) = 1;
        affine_info[1].A.at<double>(1,2) = 0;

        affine_info[2].A.at<double>(0,0) = 1;
        affine_info[2].A.at<double>(0,2) = 0;
        affine_info[2].A.at<double>(1,1) = 1;
        affine_info[2].A.at<double>(1,2) = translation;

        affine_info[3].A.at<double>(0,0) = 1;
        affine_info[3].A.at<double>(0,2) = 0;
        affine_info[3].A.at<double>(1,1) = 1;
        affine_info[3].A.at<double>(1,2) = -translation;

        affine_info[4].A = cv::getRotationMatrix2D(center, angle, 1);
        affine_info[5].A = cv::getRotationMatrix2D(center, -angle, 1);

//        affine_info[6].A = cv::getRotationMatrix2D(center, 0, pscale);
//        affine_info[7].A = cv::getRotationMatrix2D(center, 0, nscale);

        affine_info[6].A.at<double>(0,0) = 1;
        affine_info[6].A.at<double>(1,1) = 1;
    }

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {
        // options and parameters
        w = rf.check("w", Value(640)).asInt64();
        h = rf.check("h", Value(480)).asInt64();
        eros_k = rf.check("eros_k", Value(21)).asInt32();
        eros_d = rf.check("eros_d", Value(0.1)).asFloat64();
        period = rf.check("period", Value(0.01)).asFloat64();
        init_filter_width = rf.check("filter_w", Value(73)).asInt64();
        init_filter_height = rf.check("filter_h", Value(51)).asInt64();
        detection_thresh = rf.check("thresh", Value(20000)).asInt64();

        // module name
        setName((rf.check("name", Value("/object_position")).asString()).c_str());

        eros.init(w, h, eros_k, eros_d);

        if (!input_port.open("/object_position/AE:i")){
            yError()<<"cannot open input port";
            return false;
        }

//        yarp::os::Network::connect(input_port.getName(), getName("/AE:i"), "fast_tcp");
        yarp::os::Network::connect("/file/leftdvs:o", "/object_position/AE:i", "fast_tcp");

        roi_full = cv::Rect(0,0,640,480);
        detection_roi = cv::Rect(270, 190, 100, 100);
        detector.initialize(init_filter_width, init_filter_height, detection_roi, detection_thresh); // create and visualize filter for detection phase

        tracking_bool = false;
        cv::namedWindow("tracking", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("tracking", cv::Size(w,h));
        cv::moveWindow("tracking", 0,0);
        cv::moveWindow("affines", 640,0);
        cv::moveWindow("eros tracked", 740,0);

        createAffines(translation, cv::Point2f(detection_roi.width/2, detection_roi.height/2), angle, pscale, nscale);
        last_position_homogenous = cv::Mat::zeros(3,1, CV_64F);
        last_position_homogenous.at<double>(2,0) = 1;

        black_image = cv::Mat::zeros(h, w, CV_8U);

        affine_thread = std::thread([this]{fixed_step_loop();});

        return true;
    }

    void fixed_step_loop() {

        while (!input_port.isStopping()) {
            ev::info my_info = input_port.readChunkT(0.004, true);
//            yInfo()<<my_info.count<<my_info.duration<<my_info.timestamp;
            for (auto &v : input_port) {
                eros.update(640 - v.x, 480 - v.y);
            }

            cv::GaussianBlur(eros.getSurface(), eros_filtered, cv::Size(5, 5), 0);

            if (!tracking_bool) {
                if (detector.detect(eros_filtered)) { //if the maximum of the convolution is greater than a threshold

                    tracking_bool = true;
                    eros_filtered.copyTo(eros_detected);
                    detection_roi = cv::Rect(detector.max_loc.x - detection_roi.width / 2,
                                             detector.max_loc.y - detection_roi.height / 2, detection_roi.width,
                                             detection_roi.height);
                    eros_detected_roi = eros_detected(detection_roi);

                    last_template = eros_detected_roi;
                    new_template = last_template;
                    last_roi = detection_roi;

                    last_position_homogenous.at<double>(0, 0) = detector.max_loc.x - last_roi.x;
                    last_position_homogenous.at<double>(1, 0) = detector.max_loc.y - last_roi.y;

                    yInfo() << "detected puck position = (" << detector.max_loc.x << detector.max_loc.y << ")";
                }
            } else {
                eros_filtered.copyTo(eros_tracked);
                eros_tracked_roi = eros_tracked(last_roi);
                cv::imshow("eros tracked", eros_tracked_roi);

                cv::Mat h_prev, h_curr, h_new;
                for (int affine = 0; affine < affine_info.size(); affine++) {
                    if (affine < 4)
                        cv::warpAffine(last_template, affine_info[affine].warped_img, affine_info[affine].A,
                                       affine_info[affine].warped_img.size());
                    else
                        cv::warpAffine(new_template, affine_info[affine].warped_img, affine_info[affine].A,
                                       affine_info[affine].warped_img.size());

                    cv::Rect intersection_roi = last_roi & roi_full;
                    cv::Rect centered_roi(black_image.cols/2-detection_roi.width/2, black_image.rows/2-detection_roi.height/2, detection_roi.width, detection_roi.height);
                    cv::Rect centered_roi_intersection(black_image.cols/2-intersection_roi.width/2, black_image.rows/2-intersection_roi.height/2, intersection_roi.width, intersection_roi.height);
                    affine_info[affine].warped_img.copyTo(black_image(centered_roi));
                    cv::Mat new_image = black_image(centered_roi_intersection);

                    affine_info[affine].score = similarity_score(eros_tracked_roi, new_image);
                    scores_vector.push_back(affine_info[affine].score);
                    if (affine==0)
                        h_prev = affine_info[affine].warped_img;
                    else if (affine == 1)
                        h_curr = affine_info[affine].warped_img;
                    else{
                        h_curr = affine_info[affine].warped_img;
                        cv::vconcat(h_prev, h_curr, h_new);
                        h_prev = h_new;
                    }
                }

                cv::imshow("affines", h_new);

                new_position = affine_info[max_element(scores_vector.begin(), scores_vector.end()) - scores_vector.begin()].A * last_position_homogenous;
                new_roi = cv::Rect(new_position.at<double>(0, 0) + last_roi.x - detection_roi.width / 2,
                                   new_position.at<double>(1, 0) + last_roi.y - detection_roi.height / 2,
                                   detection_roi.width, detection_roi.height) & roi_full;
                new_template = affine_info[max_element(scores_vector.begin(), scores_vector.end()) -
                                           scores_vector.begin()].warped_img;

                yInfo() << scores_vector;
                yInfo() << "highest score =" << *max_element(scores_vector.begin(), scores_vector.end())
                        << max_element(scores_vector.begin(), scores_vector.end()) - scores_vector.begin();
                yInfo() << "old position = (" << last_position_homogenous.at<double>(0, 0)
                        << last_position_homogenous.at<double>(1, 0) << "), new position = ("
                        << new_position.at<double>(0, 0) << new_position.at<double>(1, 0) << ")";

                last_position_homogenous.at<double>(0, 0) = new_position.at<double>(0, 0);
                last_position_homogenous.at<double>(1, 0) = new_position.at<double>(1, 0);
                last_position_homogenous.at<double>(2, 0) = 1;
                last_roi = new_roi;
                last_template = new_template;

                scores_vector.clear();

//                cv::ellipse(eros_tracked, cv::Point(new_position.at<double>(0,0)+new_roi.x, new_position.at<double>(1,0)+new_roi.y), Size(init_filter_width/2,init_filter_height/2),0,0,360, 255, 2);
                cv::circle(eros_tracked, cv::Point(new_position.at<double>(0,0)+new_roi.x, new_position.at<double>(1,0)+new_roi.y), 2, 255, -1);
                imshow("tracking", eros_tracked);
                cv::waitKey(0);

            }
        }
    }

    bool updateModule(){
        return true;
    }

    double getPeriod() override{
        return period;
    }

    bool interruptModule() override {
        return true;
    }

    bool close() override {

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
    objectPos objectpos;

    /* run the module: runModule() calls configure first and, if successful, it then runs */
    return objectpos.runModule(rf);
}