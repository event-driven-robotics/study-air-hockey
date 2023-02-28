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

    std::array<affine_bundle, 9> affine_info;
    double translation{2}, angle{0.25}, pscale{1.02}, nscale{0.98};
    std::vector<cv::Mat> affines_vector;
    std::vector<double> scores_vector;
    cv::Mat initial_template, last_template, rot_template, rot_tr_template;
    cv::Point initial_position;
    cv::Point new_position;

    cv::Mat black_image;
    cv::Rect square;
    cv::Mat square_template, square_template_bgr, square_template_gray;
    bool first_it{true};
    bool tracking_bool{false};
    double width_roi, height_roi;
    cv::Rect roi_around_shape, roi_around_eros;

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

        affine_info[6].A = cv::getRotationMatrix2D(center, 0, pscale);
        affine_info[7].A = cv::getRotationMatrix2D(center, 0, nscale);

        affine_info[8].A.at<double>(0,0) = 1;
        affine_info[8].A.at<double>(1,1) = 1;
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
        eros_k = rf.check("eros_k", Value(17)).asInt32();
        eros_d = rf.check("eros_d", Value(0.3)).asFloat64();
        period = rf.check("period", Value(0.01)).asFloat64();
        init_filter_width = rf.check("filter_w", Value(73)).asInt64();
        init_filter_height = rf.check("filter_h", Value(55)).asInt64();
        detection_thresh = rf.check("thresh", Value(10000)).asInt64();
        width_roi = rf.check("w_roi", Value(150)).asFloat64();
        height_roi = rf.check("h_roi", Value(150)).asFloat64();

        // module name
        setName((rf.check("name", Value("/object_position")).asString()).c_str());

        eros.init(w, h, eros_k, eros_d);

        if (!input_port.open("/object_position/AE:i")){
            yError()<<"cannot open input port";
            return false;
        }

//        yarp::os::Network::connect("/file/leftdvs:o", "/object_position/AE:i", "fast_tcp");
        yarp::os::Network::connect("/atis3/AE:o", "/object_position/AE:i", "fast_tcp");


//        cv::namedWindow("tracking", cv::WINDOW_AUTOSIZE);
//        cv::resizeWindow("tracking", cv::Size(w,h));
//        cv::moveWindow("tracking", 0,0);
//        cv::moveWindow("affines", 640,0);
//        cv::moveWindow("eros tracked", 740,0);


        // CREATE THE B&W TEMPLATE
        black_image = cv::Mat::zeros(h, w, CV_8U);

        //DRAW THE STAR
//        cv::line(black_image, cv::Point(327,172), cv::Point(336,204), cv::Scalar(255),4,8,0);
//        cv::line(black_image, cv::Point(336,204), cv::Point(367,208), cv::Scalar(255),4,8,0);
//        cv::line(black_image, cv::Point(367,208), cv::Point(341,226), cv::Scalar(255),4,8,0);
//        cv::line(black_image, cv::Point(341,226), cv::Point(348,258), cv::Scalar(255),4,8,0);
//        cv::line(black_image, cv::Point(348,258), cv::Point(324,237), cv::Scalar(255),4,8,0);
//        cv::line(black_image, cv::Point(324,237), cv::Point(296,256), cv::Scalar(255),4,8,0);
//        cv::line(black_image, cv::Point(296,256), cv::Point(308,225), cv::Scalar(255),4,8,0);
//        cv::line(black_image, cv::Point(308,225), cv::Point(283,203), cv::Scalar(255),4,8,0);
//        cv::line(black_image, cv::Point(283,203), cv::Point(316,202), cv::Scalar(255),4,8,0);
//        cv::line(black_image, cv::Point(316,202), cv::Point(327,172), cv::Scalar(255),4,8,0);
//
//        initial_position.x = 325;
//        initial_position.y = 219;

        // DRAW A SQUARE
        square = cv::Rect(306,263, 87, 93);
        cv::rectangle(black_image, square, cv::Scalar(255),4,8,0); // if you draw a rectangle... is not saved in the matrix numbers

        initial_position.x = square.x + square.width/2;
        initial_position.y = square.y + square.height/2;

        roi_around_shape = cv::Rect2d(initial_position.x-width_roi/2, initial_position.y-height_roi/2, width_roi, height_roi);
        roi_around_eros = roi_around_shape;

        black_image.convertTo(square_template, CV_64F);
        initial_template = square_template(roi_around_shape);
        last_template = initial_template;

        createAffines(translation, initial_position, angle, pscale, nscale);

        // to detect the circle on slider
//        detection_roi = cv::Rect(220, 140, 100, 100);
//        detector.initialize(init_filter_width, init_filter_height, detection_roi, detection_thresh);

        affine_thread = std::thread([this]{fixed_step_loop();});

        return true;
    }

    void fixed_step_loop() {

        double sum_tx{0}, sum_ty{0}, sum_rot{0}, sum_scale{1};

        while (!input_port.isStopping()) {
            ev::info my_info = input_port.readChunkT(0.001, true);
//            yInfo()<<my_info.count<<my_info.duration<<my_info.timestamp;
            for (auto &v : input_port) {
//                if(v.x>roi_around_eros.x && v.x<roi_around_eros.x+roi_around_eros.width && v.y >roi_around_eros.y && v.y<roi_around_eros.y+roi_around_eros.height)
                    eros.update(v.x, v.y);
//                eros.update(640-v.x, 480-v.y); //when output from zynqGrabber
            }

            cv::GaussianBlur(eros.getSurface(), eros_filtered, cv::Size(5, 5), 0);
            eros_filtered(roi_around_eros).copyTo(eros_tracked); // is 0 type CV_8UC1
            eros_tracked.convertTo(eros_tracked_64f, CV_64F);  // is 6 type CV_64FC1

            for (int affine = 0; affine < affine_info.size(); affine++) {
                // warp on independent axes the roto-translated template
                cv::warpAffine(last_template, affine_info[affine].warped_img, affine_info[affine].A,
                               affine_info[affine].warped_img.size());

                if(affine == 6){
                    double n_white_pix_initial_template = cv::countNonZero(initial_template);
                    double sum_intensities_initial_template = 255*n_white_pix_initial_template;
                    double n_white_pix_rot_template = cv::countNonZero(affine_info[affine].warped_img);
                    cv::Mat new_rot_template;
                    cv::threshold(affine_info[affine].warped_img, new_rot_template, 0, 255, THRESH_BINARY);
                    double scaling_factor = n_white_pix_initial_template/n_white_pix_rot_template;
//                    yInfo()<<n_white_pix_initial_template<<n_white_pix_rot_template<<sum_intensities_initial_template<<scaling_factor;
                    cv::Mat scaled_template;
                    affine_info[affine].warped_img = scaling_factor*new_rot_template;
                }

                affine_info[affine].score = similarity_score(eros_tracked_64f, affine_info[affine].warped_img);
                scores_vector.push_back(affine_info[affine].score);
//                cv::imshow("affine"+std::to_string(affine), affine_info[affine].warped_img);
            }

            int best_score_index = max_element(scores_vector.begin(), scores_vector.end()) - scores_vector.begin();
            double best_score = *max_element(scores_vector.begin(), scores_vector.end());

//            yInfo() << scores_vector;
//            yInfo() << "highest score =" << best_score_index << best_score;

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
            else if (best_score_index == 6)
                sum_scale = sum_scale*pscale;
            else if (best_score_index == 7)
                sum_scale = sum_scale*nscale;

//            yInfo()<<"Sum ="<<sum_tx<<sum_ty<<sum_rot<<sum_scale;

            //warp the initial template by the affine state only to rotate
//            cv::Mat rotMat = updateRotMat(sum_rot, 1, cv::Point2f(0,0));
            // rot template obtained by warping for an affine including rotation and scale
            cv::Mat rotMatfunc = getRotationMatrix2D(cv::Point2d(initial_position.x-roi_around_shape.x, initial_position.y-roi_around_shape.y), sum_rot, sum_scale);
            cv::warpAffine(initial_template, rot_template, rotMatfunc, rot_template.size());
//            cv::circle(rot_template, cv::Point(initial_position.x-roi_around_shape.x, initial_position.y-roi_around_shape.y),1,255,1,8,0);
            imshow("template", rot_template);

//            if(sum_scale>1 && scaled_factor>1){
//                cv::Mat scaled_template;
//                scaled_template = cv::Mat::zeros(rot_template.cols, rot_template.rows, CV_64F);
//                scaled_template = rot_template/scaled_factor;
//                printMatrix(scaled_template);
//                imshow("scaled template", scaled_template);
//
//                scaled_template.copyTo(rot_template);
//            }

//            cv::Mat trMat =  updateTrMat(sum_tx, sum_ty);
//            cv::warpAffine(rot_template, rot_tr_template, trMat, rot_tr_template.size());
//            imshow("roto-translated template", rot_tr_template);

            new_position = cv::Point2d(initial_position.x+sum_tx,initial_position.y+sum_ty);
//            yInfo() << "new position = ("<<new_position.x<<new_position.y<<")";
            last_template = rot_template;
            roi_around_eros = cv::Rect2d(new_position.x-width_roi/2, new_position.y-height_roi/2, width_roi, height_roi);

            scores_vector.clear();

//            cv::circle(eros_tracked, cv::Point2d(initial_position.x-roi_around_shape.x, initial_position.y-roi_around_shape.y), 2, 255, -1);
            imshow("EROS ROI", eros_tracked);
            cv::circle(eros_filtered, new_position, 2, 255, -1);
            cv::rectangle(eros_filtered, roi_around_eros, 255,1,8,0);
            imshow("EROS", eros_filtered);
            cv::waitKey(1);

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