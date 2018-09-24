/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file common.hpp
 * \brief Common functions for GluonCV cpp inference demo
 * \author Joshua Zhang
 */
#include "mxnet-cpp/MxNetCpp.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <map>
#include <cmath>
#include <random>
#include <iomanip>

using namespace mxnet::cpp;

// resize short within
inline cv::Mat ResizeShortWithin(cv::Mat src, int short_size, int max_size, int mult_base) {
    double h = src.rows;
    double w = src.cols;
    double im_size_min = h;
    double im_size_max = w;
    if (w < h) {
        im_size_min = w;
        im_size_max = h;
    }
    double mb = mult_base;  // this is the factor of the output shapes
    double scale = static_cast<double>(short_size) / static_cast<double>(im_size_min);
    if ((std::round(scale * im_size_max / mb) * mb) > max_size) {
        // fit in max_size
        scale = std::floor(static_cast<double>(max_size) / mb) * mb / im_size_max;
    }
    int new_w = static_cast<int>(std::round(w * scale / mb) * mb);
    int new_h = static_cast<int>(std::round(h * scale / mb) * mb);
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(new_w, new_h));
    return dst;
}

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
inline NDArray AsData(std::string filename, Context ctx = Context::cpu()) {
    cv::Mat bgr_image = cv::imread(filename, 1);
    return AsData(bgr_image, ctx);
}

inline void LoadCheckpoint(const std::string prefix, const unsigned int epoch,
                           Symbol* symbol, std::map<std::string, NDArray>* arg_params,
                           std::map<std::string, NDArray>* aux_params,
                           Context ctx = Context::cpu()) {
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
            args[name] = iter.second.Copy(ctx);
        else if (type == "aux:")
            auxs[name] = iter.second.Copy(ctx);
        else
            continue;
    }
    NDArray::WaitAll();

    *symbol = new_symbol;
    *arg_params = args;
    *aux_params = auxs;
}

inline bool EndsWith(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline std::vector<std::string> LoadClassNames(std::string filename) {
    std::vector<std::string> classes;
    std::string line;
    std::ifstream infile(filename);
    while(infile >> line) {
        classes.emplace_back(line);
    }
    return classes;
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

inline void PutLabel(cv::Mat &im, const std::string label, const cv::Point & orig, cv::Scalar color) {
    int fontface = cv::FONT_HERSHEY_DUPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;
    double alpha = 0.6;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    // make sure roi inside image region
    cv::Rect blend_rect = cv::Rect(orig + cv::Point(0, baseline),
        orig + cv::Point(text.width, -text.height)) & cv::Rect(0, 0, im.cols, im.rows);
    cv::Mat roi = im(blend_rect);
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
        LOG(INFO) << "Start Ploting with visualize score threshold: " << thresh;
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
