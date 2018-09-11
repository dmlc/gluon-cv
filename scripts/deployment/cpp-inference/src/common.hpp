#include "mxnet-cpp/MxNetCpp.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <cmath>
#include <random>
#include <iomanip>

using namespace mxnet::cpp;

// Load data from CV BGR image
inline NDArray AsData(cv::Mat bgr_image, Context ctx = Context::cpu()) {
    // convert BGR image from OpenCV to RGB in MXNet.
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
    // convert to float32 from uint8
    rgb_image.convertTo(rgb_image, CV_32FC3);
    // flatten to single channel, and single row.
    cv::Mat flat_image = rgb_image.reshape(1, 1);
    // a vector of raw pixel values, no copy
    std::vector<float> data_buffer;
    data_buffer.insert(
        data_buffer.end(), 
        flat_image.ptr<float>(0), 
        flat_image.ptr<float>(0) + flat_image.cols);
    // construct NDArray from data buffer
    return NDArray(data_buffer, Shape(1, rgb_image.rows, rgb_image.cols, 3), ctx);
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
    ss << std::setw(4) << std::setfill('0') << epoch;
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
// convert color from hsv to bgr for plotting
inline cv::Scalar HSV2BGR(cv::Scalar hsv) {
    cv::Mat from(1, 1, CV_32FC3, hsv);
    cv::Mat to;
    cv::cvtColor(from, to, cv::COLOR_HSV2BGR);
    auto pixel = to.at<cv::Vec3f>(0, 0);
    unsigned char b = static_cast<unsigned char>(pixel[0] * 255);
    unsigned char g = static_cast<unsigned char>(pixel[1] * 255);
    unsigned char r = static_cast<unsigned char>(pixel[2] * 255);
    return cv::Scalar(b, g, r);
}

void PutLabel(cv::Mat &im, const std::string label, const cv::Point & orig, cv::Scalar color) {
    int fontface = cv::FONT_HERSHEY_DUPLEX;
    double scale = 0.6;
    int thickness = 1;
    int baseline = 0;
    double alpha = 0.6;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::Mat roi = im(cv::Rect(orig + cv::Point(0, baseline), orig + cv::Point(text.width, -text.height)));
    cv::Mat blend(roi.size(), CV_8UC3, color);
    // cv::rectangle(im, orig + cv::Point(0, baseline), orig + cv::Point(text.width, -text.height), CV_RGB(0, 0, 0), CV_FILLED);
    cv::addWeighted(blend, alpha, roi, 1.0 - alpha, 0.0, roi);
    cv::putText(im, label, orig, fontface, scale, cv::Scalar(255, 255, 255), thickness, 8);
}

// plot bounding boxes on raw image
inline cv::Mat PlotBbox(cv::Mat img, NDArray bboxes, NDArray scores, NDArray labels, 
               float thresh, std::vector<std::string> class_names, 
               std::map<int, cv::Scalar> colors, bool verbose) {
    int num = bboxes.GetShape()[1];
    std::mt19937 eng;
    std::uniform_real_distribution<float> rng(0, 1);
    float hue = rng(eng);
    bboxes.WaitToRead();
    scores.WaitToRead();
    labels.WaitToRead();
    if (verbose) {
        LOG(INFO) << "Start Ploting, visualize score threshold: " << thresh << "...";
    }
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
                colors[cls_id] = HSV2BGR(cv::Scalar(hue * 255, 0.75, 0.95));
            } else {
                // generate color for this id
                hue += 0.618033988749895;  // golden ratio
                hue = fmod(hue, 1.0);
                LG << "hue: " << hue;
                colors[cls_id] = HSV2BGR(cv::Scalar(hue * 255, 0.75, 0.95));
            }
        }

        // draw bounding box
        auto color = colors[cls_id];
        cv::Point pt1(bboxes.At(0, i, 0), bboxes.At(0, i, 1));
        cv::Point pt2(bboxes.At(0, i, 2), bboxes.At(0, i, 3));
        cv::rectangle(img, pt1, pt2, color, 2);

        if (verbose) {
            if (cls_id >= class_names.size()) {
                LOG(INFO) << "id: " << cls_id << ", scores: " << score;
            } else {
                LOG(INFO) << "id: " << class_names[cls_id] << ", scores: " << score;
            }
            
        }

        // put text
        std::string txt;
        if (class_names.size() > cls_id) {
            txt += class_names[cls_id];
        }
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << score;
        txt += " " + ss.str();
        // cv::putText(img, txt, cv::Point(pt1.x, pt1.y - 5), , 0.6, color, 1);
        PutLabel(img, txt, pt1, color);
    }
    return img;
}
}  // namespace viz