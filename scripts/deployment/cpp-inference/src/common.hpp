#include "mxnet-cpp/MxNetCpp.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <iomanip>
#include <stringstream>
#include <map>
#include <cmath>

using namespace mxnet::cpp;

// Load data from CV BGR image
inline NDArray AsData(cv::Mat bgr_image) {
    // convert BGR image from OpenCV to RGB in MXNet.
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    // convert to float32 from uint8
    rgb_image.convertTo(rgb_image, CV_32FC3);
    // flatten to single channel, and single row.
    cv::Mat flat_image = rgb_image.reshape(1, 1);
    // a vector of raw pixel values, no copy
    std::vector<float> data_buffer;
    data_buffer.insert(
        data_buffer.end(), 
        flat_image.ptr<float>(i), 
        flat_image.ptr<float>(i) + flat_image.cols);
    // construct NDArray from data buffer
    data = NDArray(data_buffer, Shape(1, rgb_image.rows, rgb_image.cols, 3), ctx);
    return data;
}

// Load data from filename
inline NDArray AsData(std::string filename) {
    cv::Mat bgr_image = cv::imread(filename, 1);
    return AsData(bgr_image);
}

inline void LoadCheckpoint(const std::string prefix, const unsigned int epoch, 
                           Symbol* symbol, std::map<std::string, NDArray>* arg_params,
                           std::map<std::string, NDArray>* aux_params) {
    // load symbol from JSON
    Symbol new_symbol = Symbol::Load(prefix + "-symbol.json");
    // load parameters
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << i;
    std::string filepath = prefix + "-" + ss.str() + ".params";
    std::map<std::string, NDArray> params = NDArray::LoadToMap(filepath);
    std::map<std::string, NDArray> args;
    std::map<std::string, NDArray> auxs;
    for (auto iter : params) {
        std::string type = iter.first.substr(0, 4);
        std::string name = iter.first.substr(4);
        if (type == "arg:")
            args[name] = iter.second;
        else if (type == "aux:")
            auxs[name] = iter.second;
        else
            continue;
    }

    *symbol = new_symbol;
    *arg_params = args;
    *aux_params = auxs;
}

namespace viz {
void cv::Scalar HSV2BGR(cv::Scalar<float> hsv) {
    cv::Mat from(1, 1, CV_32FC3, hsv);
    cv::Mat to;
    cv::cvtColor(from, to, cv::COLOR_HSV2BGR);
    auto pixel = to.at<cv::Vec3b>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}
// plot bounding boxes on raw image
void plot_bbox(cv::Mat img, NDArray bboxes, NDArray scores, NDArray labels, 
               float thresh = 0.3, std::vector<std::string> > class_names = {}, 
               std::map<int, cv::Scalar> colors = {}, bool verbose = False) {
    int num = bboxes.GetShape()[1];
    float hue = 0;
    for (int i = 0; i < num; ++i) {
        float score = scores.At(0, 0, i);
        float label = labels.At(0, 0, i);
        if (score < thresh) continue;
        if (label < 0) continue;

        int cls_id = static_cast<int>(label);
        if (colors.find(cls_id) == colors.end()) {
            // create a new color
            int csize = static_cast<int>(class_names.size());
            if (class_names.size() > 0) {
                float hue = label / csize;
                colors[cls_id] = HSV2BGR(cv::Scalar<float>(hue, 0.5, 0.95));
            } else {
                // generate color for this id
                hue += 0.618033988749895;  // golden ratio
                hue = fmod(hue, 1.0);
                colors[cls_id] = HSV2BGR(hue, 0.5, 0.95);
            }
        }

        // draw bounding box
        auto color = colors[cls_id];
        cv::Point pt1(bboxes.At(0, i, 0), bboxes.At(0, i, 1));
        cv::Point pt2(bboxes.At(0, i, 2), bboxes.At(0, i, 3));
        cv::rectangle(img, pt1, pt2, color);

        // put text
        std::string txt;
        if (class_names.size() > cls_id) {
            txt += class_names[cls_id];
        }
        std::stringstream ss;
        ss << fixed << std::setprecision(3) << score;
        txt += " " + ss.str();
        cv::putText(img, txt, pt1, cv::FONT_HERSHEY_TRIPLEX, color, 1.5);
    }
    return img;
}
}  // namespace viz