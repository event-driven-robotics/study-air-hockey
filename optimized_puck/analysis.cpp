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
    cv::Mat eros_filtered, eros_detected, eros_detected_roi, eros_tracked, eros_tracked_roi, eros_tracked_64f;

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
        cv::Mat warped_img;
        double score;
        double state;

    } affine_bundle;

    std::array<affine_bundle, 7> affine_info;
    double translation{2}, angle{1}, pscale{1.05}, nscale{0.95};
    std::vector<cv::Mat> affines_vector;
    std::vector<double> scores_vector;
    cv::Mat initial_template, last_template, new_template, rot_template, rot_tr_template;
    cv::Point2f initial_position;
    cv::Point2f new_position;
    cv::Rect last_roi, new_roi;

    cv::Mat black_image, white_image;
    cv::Rect first_square;
    cv::Mat square_template, square_template_bgr, square_template_gray;

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

    cv::Mat updateRotMat(double angle, double scale, cv::Point2f center){
        
        // the state is an affine
        cv::Mat rotMat = cv::Mat::zeros(2,3, CV_64F);

        double alpha = scale*cos(angle);
        double beta = scale*sin(angle);

        rotMat.at<double>(0,0) = alpha; rotMat.at<double>(0,1) = beta; rotMat.at<double>(0,2) = (1-alpha)*center.x - beta*center.y;
        rotMat.at<double>(1,0) = -beta; rotMat.at<double>(1,1) = alpha; rotMat.at<double>(1,2) = beta*center.x - (1-alpha)*center.y;

//        yInfo()<<alpha<<beta<<center.x<<center.y;

        return rotMat;
    }

    cv::Mat updateTrMat(double translation_x, double translation_y){

        // the state is an affine
        cv::Mat trMat = cv::Mat::zeros(2,3, CV_64F);

        trMat.at<double>(0,0) = 1; trMat.at<double>(0,1) = 0; trMat.at<double>(0,2) = translation_x;
        trMat.at<double>(1,0) = 0; trMat.at<double>(1,1) = 1; trMat.at<double>(1,2) = translation_y;

        return trMat;
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
        yarp::os::Network::connect("/atis3/AE:o", "/object_position/AE:i", "fast_tcp");

        roi_full = cv::Rect(0,0,640,480);
        detection_roi = cv::Rect(270, 190, 100, 100);
        detector.initialize(init_filter_width, init_filter_height, detection_roi, detection_thresh); // create and visualize filter for detection phase

        tracking_bool = false;
        cv::namedWindow("tracking", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("tracking", cv::Size(w,h));
        cv::moveWindow("tracking", 0,0);
        cv::moveWindow("affines", 640,0);
        cv::moveWindow("eros tracked", 740,0);

        black_image = cv::Mat::zeros(h, w, CV_8U);
        first_square = cv::Rect(306,263, 87, 93);
        cv::rectangle(black_image, first_square, cv::Scalar(255),4,8,0); // if you draw a rectangle... is not saved in the matrix numbers

        initial_position.x = first_square.x + first_square.width/2;
        initial_position.y = first_square.y + first_square.height/2;

        cv::imwrite("/code/air_hockey_event_driven_repo/study-air-hockey/optimized_puck/square_template.jpg", black_image);

        square_template_bgr = cv::imread("/code/air_hockey_event_driven_repo/study-air-hockey/optimized_puck/square_template.jpg");
        cv::cvtColor(square_template_bgr, square_template_gray, COLOR_BGR2GRAY);
        square_template_gray.convertTo(square_template, CV_64F);
        initial_template = square_template;
        last_template = initial_template;

        createAffines(translation, initial_position, angle, pscale, nscale);

        affine_thread = std::thread([this]{fixed_step_loop();});

        return true;
    }

    void fixed_step_loop() {

        double sum_tx{0}, sum_ty{0}, sum_rot{0};

        while (!input_port.isStopping()) {
            ev::info my_info = input_port.readChunkT(0.01, true);
//            yInfo()<<my_info.count<<my_info.duration<<my_info.timestamp;
            for (auto &v : input_port) {
                eros.update(v.x, v.y);
            }

            cv::GaussianBlur(eros.getSurface(), eros_filtered, cv::Size(5, 5), 0);
            eros_filtered.copyTo(eros_tracked);
            eros_tracked.convertTo(eros_tracked_64f, CV_64F);

            for (int affine = 0; affine < affine_info.size(); affine++) {
                // warp on independent axes the roto-translated template
                cv::warpAffine(last_template, affine_info[affine].warped_img, affine_info[affine].A,
                                   affine_info[affine].warped_img.size());
                affine_info[affine].score = similarity_score(eros_tracked_64f, affine_info[affine].warped_img);
                scores_vector.push_back(affine_info[affine].score);
                cv::imshow("affine"+std::to_string(affine), affine_info[affine].warped_img);
            }

            int best_score_index = max_element(scores_vector.begin(), scores_vector.end()) - scores_vector.begin();
            double best_score = *max_element(scores_vector.begin(), scores_vector.end());

            yInfo() << scores_vector;
            yInfo() << "highest score =" << best_score_index << best_score;

            //update the state
            if (best_score_index == 0)
                sum_tx += translation;
            else if (best_score_index == 1)
                sum_tx -= translation;
            else if (best_score_index == 2)
                sum_ty += translation;
            else if (best_score_index == 3)
                sum_ty -= translation;
            else if (best_score_index == 4)
                sum_rot += angle;
            else if (best_score_index == 5)
                sum_rot -= angle;

            yInfo()<<"Sum ="<<sum_tx<<sum_ty<<sum_rot;

            //warp the initial template by the affine state only to rotate
//            cv::Mat rotMat = updateRotMat(sum_rot, 1, cv::Point2f(0,0));
            cv::Mat rotMatfunc = getRotationMatrix2D(initial_position, sum_rot, 1);
            cv::warpAffine(initial_template, rot_template, rotMatfunc, rot_template.size());
//            printMatrix(rotMat);
//            printMatrix(rotMatfunc);
            imshow("initial rotated template", rot_template);
            cv::Mat trMat =  updateTrMat(sum_tx, sum_ty);
            cv::warpAffine(rot_template, rot_tr_template, trMat, rot_tr_template.size());
//            printMatrix(trMat);
            imshow("initial rotated and translated template", rot_tr_template);

            new_position = cv::Point2f(initial_position.x+sum_tx,initial_position.y+sum_ty);
            yInfo() << "new position = ("<<new_position.x<<new_position.y<<")";
            last_template = rot_tr_template;

            scores_vector.clear();

            cv::circle(eros_tracked, new_position, 2, 255, -1);
            imshow("tracking", eros_tracked);
            cv::waitKey(0);
            yInfo()<<" ";
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