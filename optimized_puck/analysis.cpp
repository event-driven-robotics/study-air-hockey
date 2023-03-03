#include <yarp/os/all.h>
using namespace yarp::os;

#include "eros.h"
#include "tracking.h"

class objectPos: public yarp::os::RFModule
{
private:
    int w, h;
    double period{0.1};
    int blur{21};

    ev::window<ev::AE> input_port;

    ev::EROS eros;
    int eros_k;
    double eros_d;
    cv::Mat eros_filtered, eros_detected, eros_detected_roi, eros_tracked, eros_tracked_roi, eros_tracked_32f, eros_tracked_64f;

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
    double translation{2}, angle{1.0}, pscale{1.01}, nscale{0.99};
    std::array<double, 4> state;
    std::vector<cv::Mat> affines_vector;
    std::vector<double> scores_vector;
    cv::Mat initial_template, roi_template, roi_template_64f, mexican_template, mexican_template_64f;
    cv::Point initial_position;
    cv::Point new_position;

    cv::Mat black_image;
    cv::Rect square;
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

    cv::Mat updateTrMat(double translation_x, double translation_y){

        // the state is an affine
        cv::Mat trMat = cv::Mat::zeros(2,3, CV_64F);

        trMat.at<double>(0,0) = 1; trMat.at<double>(0,1) = 0; trMat.at<double>(0,2) = translation_x;
        trMat.at<double>(1,0) = 0; trMat.at<double>(1,1) = 1; trMat.at<double>(1,2) = translation_y;

        return trMat;
    }

    void make_template(const cv::Mat &input, cv::Mat &output) {
        static cv::Mat  pos_hat, neg_hat; //canny_img, f,
        static cv::Size pblur(blur, blur);
        static cv::Size nblur(2*blur-1, 2*blur-1);
        static double minval, maxval;

        //cv::GaussianBlur(input, input, cv::Size(3, 3), 0);
        //cv::normalize(input, input, 0, 255, cv::NORM_MINMAX);
        //cv::Sobel(input,

//        input.convertTo(canny_img, CV_32F, 0.003921569);
//        cv::Sobel(canny_img, f, CV_32F, 1, 1);
//        f = (cv::max(f, 0.0) + cv::max(-f, 0.0));
//        cv::minMaxLoc(f, &minval, &maxval);
//        cv::threshold(f, f, maxval*0.05, 0, cv::THRESH_TRUNC);

        // cv::imshow("temp", f+0.5);
        //f.copyTo(output);
        // return;
        // cv::Canny(input, canny_img, canny_thresh, canny_thresh*canny_scale, 3);
        // canny_img.convertTo(f, CV_32F);

        cv::GaussianBlur(input, pos_hat, pblur, 0);
        cv::GaussianBlur(input, neg_hat, nblur, 0);
        output = pos_hat - neg_hat;

        cv::minMaxLoc(output, &minval, &maxval);
        double scale_factor = 1.0 / (2 * std::max(fabs(minval), fabs(maxval)));
        output *= scale_factor;
    }

    cv::Mat createDynamicTemplate(std::array<double, 4> state){

        cv::Mat rot_scaled_template, rot_scaled_tr_template;

        cv::Mat rotMatfunc = getRotationMatrix2D(initial_position, state[2], state[3]);
        cv::warpAffine(initial_template, rot_scaled_template, rotMatfunc, rot_scaled_template.size());
        cv::Mat trMat =  updateTrMat(state[0], state[1]);
        cv::warpAffine(rot_scaled_template, rot_scaled_tr_template, trMat, rot_scaled_tr_template.size());
        new_position = cv::Point2d(initial_position.x+state[0],initial_position.y+state[1]);

        return rot_scaled_tr_template;
    }

    void setROI(cv::Mat full_template, int buffer = 20){

        roi_around_shape = cv::boundingRect(full_template);
        roi_around_shape.x -= buffer;
        roi_around_shape.y -= buffer;
        roi_around_shape.width += buffer * 2;
        roi_around_shape.height += buffer * 2;

        roi_template = full_template(roi_around_shape);
        roi_template.convertTo(roi_template_64f, CV_64F);  // is 6 type CV_64FC1
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

        // module name
        setName((rf.check("name", Value("/object_position")).asString()).c_str());

        eros.init(w, h, eros_k, eros_d);

        if (!input_port.open("/object_position/AE:i")){
            yError()<<"cannot open input port";
            return false;
        }

//        yarp::os::Network::connect("/file/leftdvs:o", "/object_position/AE:i", "fast_tcp");
        yarp::os::Network::connect("/atis3/AE:o", "/object_position/AE:i", "fast_tcp");

        state[0]=0; state[1]=0; state[2]=0; state[3]=1;

        // CREATE THE B&W TEMPLATE
        black_image = cv::Mat::zeros(h, w, CV_8UC1);

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
        cv::rectangle(black_image, square, cv::Scalar(255),1,8,0); // if you draw a rectangle... is not saved in the matrix numbers

        initial_position.x = square.x + square.width/2;
        initial_position.y = square.y + square.height/2;

//        black_image.convertTo(initial_template, CV_32F);
        initial_template = black_image;

        createAffines(translation, initial_position, angle, pscale, nscale);

        affine_thread = std::thread([this]{fixed_step_loop();});

        return true;
    }

    void fixed_step_loop() {

        while (!input_port.isStopping()) {

            // 1) create 1 dynamic template
//            yInfo()<<"State ="<<state[0]<<state[1]<<state[2]<<state[3];
            cv::Mat dynamic_template = createDynamicTemplate(state);

            // 2) find the roi (getboundingbox function opencv)
            setROI(dynamic_template);

            cv::Point2d new_center(roi_around_shape.width/2, roi_around_shape.height/2);
            affine_info[4].A = cv::getRotationMatrix2D(new_center, angle, 1);
            affine_info[5].A = cv::getRotationMatrix2D(new_center, -angle, 1);

            affine_info[6].A = cv::getRotationMatrix2D(new_center, 0, pscale);
            affine_info[7].A = cv::getRotationMatrix2D(new_center, 0, nscale);

            // ????? resize needed -> look at 6dof

            // 3) mexican hat
            make_template(roi_template_64f, mexican_template);
            mexican_template.convertTo(mexican_template_64f, CV_64F);  // is 6 type CV_64FC1

            // 4) create 9 templates (warped images)
            for (int affine = 0; affine < affine_info.size(); affine++) {
//                yInfo()<<roi_template.cols<<roi_template.rows<<affine_info[affine].warped_img.cols<<affine_info[affine].warped_img.rows;
//                cv::warpAffine(roi_template_64f, affine_info[affine].warped_img, affine_info[affine].A, roi_template_64f.size());
                cv::warpAffine(mexican_template_64f, affine_info[affine].warped_img, affine_info[affine].A, mexican_template_64f.size());
            }

            // 5) get EROS ROI
            ev::info my_info = input_port.readChunkT(0.004, true);
            for (auto &v : input_port) {
                eros.update(v.x, v.y);
            }
            cv::GaussianBlur(eros.getSurface(), eros_filtered, cv::Size(5, 5), 0);
//            yInfo()<<roi_around_shape.x<<roi_around_shape.y<<roi_around_shape.width<<roi_around_shape.height;
            eros_filtered(roi_around_shape).copyTo(eros_tracked); // is 0 type CV_8UC1
            eros_tracked.convertTo(eros_tracked_64f, CV_64F);  // is 6 type CV_64FC1

            // 6) compare 9 templates vs eros
            for (int affine = 0; affine < affine_info.size(); affine++) {
//                yInfo()<<eros_tracked_64f.type()<<affine_info[affine].warped_img.type();
//                yInfo()<<eros_tracked_64f.cols<<eros_tracked_64f.rows<<affine_info[affine].warped_img.cols<<affine_info[affine].warped_img.rows;
                affine_info[affine].score = similarity_score(eros_tracked_64f, affine_info[affine].warped_img);
                scores_vector.push_back(affine_info[affine].score);
                cv::imshow("affine"+std::to_string(affine), affine_info[affine].warped_img);
            }

            // 7) get the maximum score
            int best_score_index = max_element(scores_vector.begin(), scores_vector.end()) - scores_vector.begin();
            double best_score = *max_element(scores_vector.begin(), scores_vector.end());
            //yInfo() << scores_vector;
//            yInfo() << "highest score =" << best_score_index << best_score;
            scores_vector.clear();

            // 8) update the state
            if (best_score_index == 0)
                state[0] += translation;
            else if (best_score_index == 1)
                state[0] -= translation;
            else if (best_score_index == 2)
                state[1] += translation;
            else if (best_score_index == 3)
                state[1] -= translation;
            else if (best_score_index == 4)
                state[2] += angle;
            else if (best_score_index == 5)
                state[2] -= angle;
            else if (best_score_index == 6)
                state[3] = state[3]*pscale;
            else if (best_score_index == 7)
                state[3] = state[3]*nscale;

            // 9) visualize
            cv::Mat norm_mexican;
            cv::normalize(mexican_template_64f, norm_mexican, 1, 0, cv::NORM_MINMAX);
            imshow("MEXICAN ROI", mexican_template_64f+0.5);
            imshow("TEMPLATE ROI", roi_template);
            imshow("EROS ROI", eros_tracked);
            cv::circle(eros_filtered, new_position, 2, 255, -1);
            cv::rectangle(eros_filtered, roi_around_shape, 255,1,8,0);
            imshow("EROS FULL", eros_filtered+dynamic_template);
//            imshow("TEMPLATE FULL", dynamic_template);
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